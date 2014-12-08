/** @file MCMC.hpp
 * @brief A visualization tool for posterior distributions using MCMC.
 */
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <queue>
#include <map>

#include "CS207/SDLViewer.hpp"
#include "CS207/Util.hpp"
#include "CS207/Color.hpp"

#include "Point.hpp"
#include "Mesh.hpp"
#include "ProblemFactory.hpp"

#include <boost/math/distributions.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/signals2/signal.hpp>
#include <boost/bind/bind.hpp>
#include <boost/function.hpp>
#include <boost/ref.hpp>

#include <SDL/SDL.h>

using namespace std;
using namespace boost::math;

// TODO: use morton code to avoid linear checking of each triangles
// TODO: implement more examples and data
// TODO: comparing with more methods like HMC or multipletiy
// TODO: save demos, gif, youtube

// TODO: automate the numbers n_x, n_y in the grid adaptively or something
// TODO: show traversals of the proposal points right above or below the posterior simulation
// TODO: set the default view for viewer
// TODO: have good visuals for comparing the prior to the posterior

// create code for stuff here
typedef double value_type;
typedef unsigned size_type;
typedef std::pair<value_type, value_type> interval;
typedef std::vector<value_type> param_type;

typedef Mesh<value_type, bool, value_type> MeshType;
typedef MeshType::GraphType GraphType;
typedef MeshType::Node Node;
typedef MeshType::Triangle Triangle;
typedef std::map<Node, size_type> NodeMapType;

typedef boost::function<void (const param_type&, size_type, size_type)> MCMC_Iteration_Event;
// define constant of PI from Boost
const double PI = boost::math::constants::pi<value_type>();

value_type sleep_interval = 0.01;
value_type sleep_step = .001;
bool show_edge = true;
size_type show_num_iteration = 1;
bool show_zaxis = false;

value_type range_x_slope = 0;
value_type range_x_int = 0;
value_type range_y_slope = 0;
value_type range_y_int = 0;

template <typename RANGE>
void set_range(const RANGE& range) {
  range_x_slope = 2./(range.x.second-range.x.first);
  range_x_int = -range.x.first*range_x_slope;
  range_y_slope = 2./(range.y.second-range.y.first);
  range_y_int = -range.y.first*range_y_slope;
}

// TODO: the rounding errors so that it works for any specified ranges/# pts
// for two-dimensional parameters (x,y)
class Grid {
public:
  /** Returns a grid of the interval separated by the number of points.
    * @param[in] x [low, high] domain of x
    * @param[in] y [low, high] domain of y
    * @param[in] n_x number of linearly spaced points in interval x
    * @param[in] n_y number of linearly space points in interval y
    * @return Mesh which connects all the points in the grid.
    *
    * Complexity: O(n_x*n_y).
    */
  MeshType operator()(interval x, interval y, size_type n_x, size_type n_y) {
    MeshType mesh;
    value_type x_step = (x.second - x.first) / (n_x - 1);
    value_type y_step = (y.second - y.first) / (n_y - 1);

    for (size_type j = n_y; j > 0; --j) {
      for (size_type i = 1; i <= n_x; ++i) {
        mesh.add_node(Point((i-1)*x_step, (j-1)*y_step, 0));
        MeshType::size_type idx = mesh.num_nodes() - 1;
        if (j != n_y) {
          if (i == 1) {
            mesh.add_triangle(mesh.node(idx), mesh.node(idx - n_x), mesh.node(idx - n_x + 1));
          } else if (i == n_x) {
            mesh.add_triangle(mesh.node(idx), mesh.node(idx - 1), mesh.node(idx - n_x));
          } else {
            mesh.add_triangle(mesh.node(idx), mesh.node(idx - 1), mesh.node(idx - n_x));
            mesh.add_triangle(mesh.node(idx), mesh.node(idx - n_x), mesh.node(idx - n_x + 1));
          }
        }
      }
    }
    return mesh;
  }
};

class InTrianglePred {
  Point p0;
 public:
  InTrianglePred(param_type p) : p0({range_x_slope*p[0]+range_x_int, range_y_slope*p[1]+range_y_int, 0}) {

  }

  bool operator()(const Triangle& tri) const {
    /*
     * v2 = alpha * v0 + beta * v1, where
     * v0 = B - A, v1 = C - A, v2 = P - A
     *
     * Then we get:
     * v0v2 = alpha * (v0v0) + beta * (v0v1)
     * v1v2 = alpha * (v0v1) + beta * (v1v1)
    */
    auto v0 = tri.node(1).position() - tri.node(0).position();
    auto v1 = tri.node(2).position() - tri.node(0).position();
    auto v2 = p0 - tri.node(0).position();

    double v0v0 = dot(v0, v0);
    double v1v1 = dot(v1, v1);
    double v0v1 = dot(v0, v1);
    double v0v2 = dot(v0, v2);
    double v1v2 = dot(v1, v2);

    double alpha = (v0v1 * v1v2 - v1v1 * v0v2) / (v0v1 * v0v1 - v0v0 * v1v1);
    double beta = (v0v1 * v0v2 - v0v0 * v1v2) / (v0v1 * v0v1 - v0v0 * v1v1);
    bool result = ((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1));

    return result;
  }
};

struct NodeValueLessComparator {
  template <typename NODE>
  bool operator()(const NODE& n1, const NODE& n2) const {
    return n1.value() < n2.value();
  }
};

/** Node position function object for use in the SDLViewer. */
struct NodePosition {
  template <typename NODE>
  Point operator()(const NODE& n) {
    double z = n.value() / max_;
    return {n.position().x, n.position().y, z};
  }

  NodePosition(double max) : max_(max) { }

 private:
  double max_;
};

// A default color functor that returns white for anything it recieves
struct HeatmapColor {
  template <typename NODE>
  CS207::Color operator()(const NODE& n) {
    double val = 0.0;
    if (max_ > 0)
      val = n.value() / max_;

    val = std::min(val, 1.);
    return CS207::Color::make_heat(val);
  }

  HeatmapColor(double max) : max_(max) { }

private:
  double max_;
};

struct TrajectoryColor
{
  template <typename NODE>
  CS207::Color operator()(const NODE& n) {
    double val = n.value() / 25.;
    return CS207::Color::make_heat(val);
  }
};

/** Update the empirical probabilities at each triangle.
    */
template <typename PRED>
void update(MeshType& mesh, size_type N, PRED pred) {
  // for (auto it = mesh.triangle_begin(); it != mesh.triangle_end(); ++it) {
  for (auto tri : mesh.triangles()) {
    if (pred(tri)) {
      //triangle that node is in
      tri.value() = ((tri.value() * N) + 1) / (N + 1);
    } else {
      tri.value() = tri.value() * N / (N + 1);
    }
  }
}

/** Set each node's value to be the average of its adjacent triangles' empirical probabilities.
  */
void post_process(MeshType& mesh) {
  for (auto node : mesh.nodes()) {
    value_type total_prob = 0;
    size_type count = 0;

    for (auto tri_it = mesh.adjacent_triangle_begin(node); tri_it != mesh.adjacent_triangle_end(node); ++tri_it) {
      total_prob += (*tri_it).value();
      ++count;
    }
    node.value() = total_prob / count;
  }
}

/* Generate bivariate (independent) normal samples. */
class mvtnorm {
 public:
  
  mvtnorm(value_type mu_x, value_type mu_y, value_type sigma_x, value_type sigma_y)
    : mu_x_(mu_x), mu_y_(mu_y), sigma_x_(sigma_x), sigma_y_(sigma_y) {
  }
  
  param_type rand(const param_type& theta, size_type) {
    (void) theta;
    std::normal_distribution<value_type> distribution_1(mu_x_, sigma_x_);
    std::normal_distribution<value_type> distribution_2(mu_y_, sigma_y_);
    
    return {distribution_1(generator_), distribution_2(generator_)};
  }
  
  value_type dens(const param_type& theta_to, const param_type& theta_from, bool log_) const {
    (void) theta_from;
    value_type x = theta_to[0];
    value_type y = theta_to[1];
    value_type dens = 1/(2*PI*sigma_x_*sigma_y_) *
      exp(- 1./2. * (pow(x-mu_x_, 2)/pow(sigma_x_, 2) + pow((y-mu_y_),2)/pow(sigma_y_,2)));
    
    return log_ ? log(dens) : dens;
  }
  
 private:
  value_type mu_x_;
  value_type mu_y_;
  value_type sigma_x_;
  value_type sigma_y_;
  std::default_random_engine generator_;
};

class Posterior_Boost {
public:
  Posterior_Boost(value_type mean, value_type sd) : normal_distr_({mean, sd}) {

  }

  value_type operator()(const param_type& theta, bool log_) const {
    value_type x = theta[0];
    value_type y = theta[1];

    value_type dens = pdf(normal_distr_, x);
    dens *= pdf(normal_distr_, y);
    return log_ ? log(dens) : dens;
  }
private:
  // student_t_distribution<value_type> student_t_distr_;
  normal normal_distr_;
};

class Posterior {
 public:
  Posterior(value_type mu_x, value_type mu_y, value_type sigma_x, value_type sigma_y)
    : mu_x_(mu_x), mu_y_(mu_y), sigma_x_(sigma_x), sigma_y_(sigma_y) {
  }
  
  value_type operator()(const param_type& theta, bool log_) const {
    value_type x = theta[0];
    value_type y = theta[1];

    value_type dens = 1/(2*PI*sigma_x_*sigma_y_) *
      exp(-1./2. * (pow(x-mu_x_, 2) / pow(sigma_x_, 2) + pow(y-mu_y_-.5, 2) / pow(sigma_y_, 2)));
    dens = dens + 1/(2*PI*sigma_x_*sigma_y_) *
      exp(- 1./2. * (pow(x-mu_x_-.433, 2)/pow(sigma_x_, 2) + pow((y-mu_y_+.25),2)/pow(sigma_y_,2)));
    dens = dens + 1/(2*PI*sigma_x_*sigma_y_) *
      exp(- 1./2. * (pow(x-mu_x_+.433, 2)/pow(sigma_x_, 2) + pow((y-mu_y_+.25),2)/pow(sigma_y_,2)));  
    dens = dens/3;
    return log_ ? log(dens) : dens;
  }
 private:
  value_type mu_x_;
  value_type mu_y_;
  value_type sigma_x_;
  value_type sigma_y_;
};

class MCMC_Simulator {
 public:
  // TODO: find some way to do lexical scoping
  /** Animates frames which represent each iteration of the sampling procedure.
    * @param[in] proposal
    * @param[in] max_N
    * @param[in] data
    * @param[in,out] mesh
    * @param[in,out] viewer
    */
  template <typename PROPOSAL, typename POSTERIOR>
  void simulate(PROPOSAL& proposal, const POSTERIOR& posterior,
    const vector<value_type> initial_val, size_type max_N, size_type size) {
    // Initialize matrix of size max_N x size.
    std::vector<param_type>theta(max_N);
    theta[0] = initial_val;
  
    std::default_random_engine generator;
    std::uniform_real_distribution<value_type> runif(0, 1);

    size_type accept_count = 0;
    for (size_type i = 1; i <= max_N; ++i) {
      // theta[i-1] is the previous state, theta_star is the new(proposed) state
      param_type theta_star = proposal.rand(theta[i-1], size);
    
      value_type accept_ratio = posterior(theta_star, true) - posterior(theta[i-1], true) + proposal.dens(theta[i-1], theta_star, true) - proposal.dens(theta_star, theta[i-1], true);
      accept_ratio = std::min(1., exp(accept_ratio));
      if (runif(generator) < accept_ratio) {
        theta[i] = theta_star;
        ++accept_count;
      } else {
        theta[i] = theta[i-1];
      }
      
      //std::cout << "N: " << theta[i][0] << ", theta: " << theta[i][1] << std::endl;
      signals_(theta[i], i, accept_count);      
    }
  }

  void connect(const MCMC_Iteration_Event& func) {
    signals_.connect(func);
  }

private:
  boost::signals2::signal<void (const param_type&, size_type, size_type)> signals_;
};

/** Callback function for each frame, in this case each iteration in MCMC step */
void callback_distribution(const param_type& theta, size_type N, size_type accept_count, MeshType& mesh, CS207::SDLViewer& viewer, NodeMapType& node_map) {
  update(mesh, N, InTrianglePred(theta));
  post_process(mesh);

  auto max_node_it = std::max_element(mesh.node_begin(), mesh.node_end(), NodeValueLessComparator());
  double max_val = (*max_node_it).value();

  if (show_zaxis) {
    viewer.add_nodes(mesh.node_begin(), mesh.node_end(),HeatmapColor(max_val), node_map);
  } else {
    viewer.add_nodes(mesh.node_begin(), mesh.node_end(),
                   HeatmapColor(max_val), NodePosition(max_val), node_map);
  }

  if (show_num_iteration == 1) {
    viewer.set_label(N);
  } else if (show_num_iteration == 2) {
    viewer.set_label(accept_count / (value_type)N * 100.0);
  } else {
    viewer.set_label("");
  }
  
  CS207::sleep(sleep_interval);
}

void callback_trajectory(const param_type& theta, size_type i, size_type accept_count, GraphType& graph, CS207::SDLViewer& viewer, NodeMapType& node_map) {
  //std::cout << "i: " << i << "; graph.size(): " << graph.size() << std::endl;
  (void) accept_count;
  static std::vector<param_type> trajectory_vec;
  
  // Force the tie conditional to be true if it is the first sample.
  auto last_position = Point(theta[0]-1, theta[1]-1, 0);
  if (i != 1) {
    last_position = Point(trajectory_vec.back()[0], trajectory_vec.back()[1], 0);
  }
  if (std::tie(theta[0], theta[1]) != std::tie(last_position.x, last_position.y)) {
    if (trajectory_vec.size() >= 25) { // Remove the 5th point before adding another.
      trajectory_vec.erase(trajectory_vec.begin());
    }
    trajectory_vec.push_back(theta);
  }
  
  // Update graph.
  graph.clear();
  size_type k = 0;
  for (auto theta_k : trajectory_vec) {
    graph.add_node(Point(range_x_slope*theta_k[0]+range_x_int, range_y_slope*theta_k[1]+range_y_int, 1));
    graph.node(k).value() = (value_type)(k + 1);
    ++k;
    if (graph.size() >= 2) { // Add edge between last two points.
      graph.add_edge(graph.node(graph.size()-1), graph.node(graph.size()-2));
    }
  }
  viewer.add_nodes(graph.node_begin(), graph.node_end(), TrajectoryColor(), node_map);
  viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);

}

/** Callback function to handle keyboard events */
void callback_keyboard(SDLKey key, MeshType& mesh, CS207::SDLViewer& viewer, NodeMapType& node_map) {
  switch(key) {
    case SDLK_DOWN:
      sleep_interval += sleep_step;
      sleep_interval = sleep_interval > .05 ? .05 : sleep_interval;
      std::cout << "Framerate:" << sleep_interval << std::endl;
      break;
    case SDLK_UP:
      sleep_interval -= sleep_step;
      sleep_interval = sleep_interval < 0. ? 0. : sleep_interval;
      std::cout << "Framerate:" << sleep_interval << std::endl;
      break;
    case SDLK_e:
      std::cout << "Edge!" << std::endl;
      show_edge = !show_edge;
      std::cout << show_edge << std::endl;
      viewer.clear();
      node_map = viewer.empty_node_map(mesh);
      if (show_edge) {
        viewer.add_nodes(mesh.node_begin(), mesh.node_end(), node_map);
        viewer.add_edges(mesh.edge_begin(), mesh.edge_end(), node_map);
      }
      break;
    case SDLK_t:
      ++show_num_iteration;
      show_num_iteration = show_num_iteration % 3;
      break;
    case SDLK_f:
      show_zaxis = !show_zaxis;
      break;
    default:
      break;
  }
}

int main() {
  auto mesh = Grid()(std::make_pair(0., 2.), std::make_pair(0., 2.), 51, 51);
  GraphType graph;
  // Print out the stats
  std::cout << mesh.num_nodes() << " "
            << mesh.num_edges() << " "
            << mesh.num_triangles() << std::endl;

  // Launch the SDLViewer
  CS207::SDLViewer viewer;
  viewer.launch();

  auto node_map_dist = viewer.empty_node_map(mesh);
  auto node_map_traj = viewer.empty_node_map(graph);
  
  viewer.add_nodes(mesh.node_begin(), mesh.node_end(), node_map_dist);
  viewer.add_edges(mesh.edge_begin(), mesh.edge_end(), node_map_dist);

  MCMC_Simulator simulator;
  ProblemFactory problem_factory;
  viewer.add_keyboard_listener(boost::bind(&callback_keyboard, _1, boost::ref(mesh), boost::ref(viewer), boost::ref(node_map_dist)));

  // param_type initials {0., 0.5};
  param_type initials {100., 0.5};
  simulator.connect(boost::bind(&callback_distribution, _1, _2, _3, boost::ref(mesh), boost::ref(viewer), boost::ref(node_map_dist)));
  simulator.connect(boost::bind(&callback_trajectory, _1, _2, _3, boost::ref(graph), boost::ref(viewer), boost::ref(node_map_traj)));

  // auto proposal = mvtnorm(1., 1., .5, .5);
  // auto posterior = Posterior(1., 1., .15, .15);
  auto proposal = problem_factory.create_proposal();
  auto posterior = problem_factory.create_posterior();
  auto data_range = problem_factory.create_range();
  set_range(data_range);
  
  simulator.simulate(proposal, posterior, initials, 5e5, 2);
}