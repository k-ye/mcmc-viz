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

#include "mesh/CS207/SDLViewer.hpp"
#include "mesh/CS207/Util.hpp"
#include "mesh/CS207/Color.hpp"

#include "mesh/CS207/Point.hpp"
#include "mesh/Mesh.hpp"
#include "SpaceSearcher.hpp"
#include "ProblemFactory.hpp"

#include <boost/math/constants/constants.hpp>
#include <boost/signals2/signal.hpp>
#include <boost/bind/bind.hpp>
#include <boost/function.hpp>
#include <boost/ref.hpp>

#include <SDL/SDL.h>
#include <omp.h>

using namespace std;
using namespace boost::math;

// TODO: save demos, gif, youtube
// TODO: parallel the process of update() using the index of each triangle
// TODO: 

#define GRID_WIDTH 2.5
#define GRID_HEIGHT 2.5
#define TRAJECTORY_LENGTH 25

struct TriangleToPoint;

typedef ProblemFactory::size_type size_type;
typedef ProblemFactory::value_type value_type;
typedef ProblemFactory::param_type param_type;
// The proposal type of the stats model
typedef ProblemFactory::ProblemProposal ProblemProposal;
// The posterior type of the stats model
typedef ProblemFactory::ProblemPosterior ProblemPosterior;
typedef ProblemFactory::ProblemRange ProblemRange;
typedef ProblemFactory::ProblemInitialParams ProblemInitialParams;
// Interval type
typedef std::pair<value_type, value_type> interval;

typedef Mesh<value_type, bool, value_type> MeshType;
typedef MeshType::GraphType GraphType;
typedef MeshType::Node Node;
typedef MeshType::Triangle Triangle;
typedef std::map<Node, size_type> NodeMapType;
// Define spatial searcher type
typedef SpaceSearcher<Triangle, TriangleToPoint> SpaceSearcherType;
// EventHandler type for MCMC iteration event.
typedef boost::function<void (const std::vector<param_type>&, size_type, size_type)> MCMC_Iteration_Event;

// controls the sleeping time between each MCMC Iteration
value_type sleep_interval = 0.01;
// controls the step to increase/decrease @a sleep_interval
value_type sleep_step = .002;
// indicator to determine what to be shown in the label
size_type show_label_state = 1;
// indicator to determine if showing 3D surface/ 2D heatmap
bool show_zaxis = false;
// slope for range mapping between actual param and Mesh coordinate on X axis
value_type range_x_slope = 0;
// intercept for range mapping between actual param and Mesh coordinate on X axis
value_type range_x_int = 0;
// slope for range mapping between actual param and Mesh coordinate on Y axis
value_type range_y_slope = 0;
// intercept for range mapping between actual param and Mesh coordinate on Y axis
value_type range_y_int = 0;

/** Set up the mapping relationship between actual params and mesh coordinate.
 * @param[in] range: object specifies a (probably approximated) range of the first two dimension of the params space
 * @pre: @arange.x.first < @a range.x.second && @a range.y.first < @a range.y.second
 */
template <typename RANGE>
void set_range(const RANGE& range) {
  assert(range.x.first < range.x.second && range.y.first < range.y.second);
  range_x_slope = GRID_WIDTH / (range.x.second-range.x.first);
  range_x_int = -range.x.first*range_x_slope;
  range_y_slope = GRID_HEIGHT / (range.y.second-range.y.first);
  range_y_int = -range.y.first*range_y_slope;
}

/* Class to initialize a square mesh */
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

/* Class to initialize trajectory for sampled points. */
class Trajectory {
public:
  /*
   * @param[in] length: length of the trajectory
   */
  GraphType operator()(size_type length) {
    GraphType graph;
    size_type num_threads = omp_get_max_threads();

    for (size_type i = 0; i < num_threads; ++i) {
      for (size_type j = 0; j < length; ++j) {
        graph.add_node(Point((value_type)j / 5., (value_type)i / 2., 1.2));
        if (j > 0) {
          graph.add_edge(graph.node(graph.size()-1), graph.node(graph.size()-2));
        }
      }
    }
    
    return graph;
  }
};

/** Calculate mapped value from true params to Mesh X-cooridnate */
value_type map_x_grid_range(value_type x) {
  return range_x_slope * x + range_x_int;
}

/** Calculate mapped value from true params to Mesh Y-cooridnate */
value_type map_y_grid_range(value_type y) {
  return range_y_slope * y + range_y_int;
}

/* Functor to determine whether a point is in a triangle. */
class InTrianglePred {
  Point p0;
 public:
  InTrianglePred(param_type p) : p0({map_x_grid_range(p[0]), map_y_grid_range(p[1]), 0.}) {

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

/** Comparator to compare the value between @a n1 and @a n2. */
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

/* A default color functor that returns colors according to the frequency. */
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

/** Default color functor to return colors according to how recent the point is
  * in the trajectory.
  */
struct TrajectoryColor {
  template <typename NODE>
  CS207::Color operator()(const NODE& n) {
    switch((size_type)n.value()) {
      case 0:
        return CS207::Color::make_rgb(0., 199./255., 1.);
      case 1:
        return CS207::Color::make_rgb(0., 1., 0.);
      case 2:
        return CS207::Color::make_rgb(1., 0., 0.);
      case 3:
        return CS207::Color::make_rgb(1., 1., 0.);
      default:
        return CS207::Color::make_rgb(1., 1., 1.);
    }
  }
};

/* Update the empirical probabilities at each triangle. */
template <typename ITER, typename PRED>
void update(ITER begin, ITER end, size_type N, PRED pred) {
  (void) N;
  for (auto it = begin; it != end; ++it) {
    if (pred(*it)) {
      (*it).value() += 1;
    }
  }
}

/** Update the empirical probabilities at each triangle.
  *  Deprecated as now we are utilizing SpaceSearcher.
  */
template <typename PRED>
void update(MeshType& mesh, size_type N, PRED pred) {
  update(mesh.triangle_begin(), mesh.triangle_end(), N, pred);
}

/* Function to apply to mesh after each iteration. */
void post_process(MeshType& mesh) {
  /** Set each node's value to be the average of its adjacent triangles' empirical
    * probabilities. */
  size_type thread_id = omp_get_thread_num();
  size_type thread_num = omp_get_num_threads();

  size_type size_per_thread = mesh.num_nodes() / thread_num;
  size_type start = thread_id * size_per_thread;
  size_type end = (thread_id == thread_num - 1) ? (thread_id+1) * size_per_thread : mesh.num_nodes();

  for (size_type i = start; i < end; ++i) {
    auto node = mesh.node(i);
    value_type total_prob = 0;
    size_type count = 0;

    for (auto tri_it = mesh.adjacent_triangle_begin(node); tri_it != mesh.adjacent_triangle_end(node); ++tri_it) {
      total_prob += (*tri_it).value();
      ++count;
    }
    node.value() = total_prob / count;
  }
}


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
  void simulate(const ProblemFactory::ProblemFactory& problem_factory, size_type max_N, size_type p) {
    size_type num_threads = omp_get_max_threads();
    // Initialize matrix of size max_N x p.
    std::vector<param_type>theta((max_N+1)*num_threads);
    // create problem definition from problem factory
    ProblemProposal proposal = problem_factory.create_proposal();
    ProblemPosterior posterior = problem_factory.create_posterior();
    auto initial_val = problem_factory.create_initial_params();
    assert(initial_val.size() >= num_threads);
    
    for (size_type i = 0; i < num_threads; ++i) {
      theta[i*max_N] = initial_val[i];
    }

    std::default_random_engine generator;
    std::uniform_real_distribution<value_type> runif(0, 1);

    size_type accept_count = 0;
    
    // this is necessary because in parallel, we canot access directly to accept_count
    for (size_type i = 1; i <= max_N; ++i) {
      std::vector<size_type> accept_bit(num_threads, 0);
      std::vector<param_type> proposed_theta(num_threads);
      #pragma omp parallel 
      {
        size_type omp_id = omp_get_thread_num();
        size_type j = omp_id * max_N + i;
        // theta[i-1] is the previous state, theta_star is the new(proposed) state
        param_type theta_star = proposal.rand(theta[j-1], p);
        // calculate the acceptance ratio
        value_type accept_ratio = posterior(theta_star, true) - posterior(theta[j-1], true)
                                + proposal.dens(theta[j-1], theta_star, true) - proposal.dens(theta_star, theta[j-1], true);
        accept_ratio = std::min(1., exp(accept_ratio));
        //std::cout << accept_ratio << std::endl;
        // decide to accept or reject proposed sample point
        if (runif(generator) < accept_ratio) {
          theta[j] = theta_star;
          accept_bit[omp_id] = 1;
        }
        else {
          theta[j] = theta[j-1];
          accept_bit[omp_id] = 0;
        }
        proposed_theta[omp_id] = theta[j];
      }
      for (auto bit : accept_bit) accept_count += bit;
      // fire the event to tell the handler that this iteration has completed
      signals_(proposed_theta, i, accept_count);      
    }
  }

  /* Connect an event listener to the MCMC_Iteration event. */
  void add_mcmc_iteration_listener(const MCMC_Iteration_Event& func) {
    signals_.connect(func);
  }

private:
  /* Event listener containers */
  boost::signals2::signal<void (const std::vector<param_type>&, size_type, size_type)> signals_;
};

/* Functor to map a triangle to a point for Morton Code. */
struct TriangleToPoint {
  template <typename TRIANGLE>
  Point operator()(const TRIANGLE& tri) const {
    Point center(0, 0, 0);
    for (size_type i = 0; i < 3; ++i) {
      center += tri.node(i).position();
    }
    center /= 3.;
    return center;
  }
};

/* Event handler of Mesh distribution display for MCMC_Iteration_Event */
void callback_distribution(const std::vector<param_type>& theta, size_type N, size_type accept_count, 
  MeshType& mesh, CS207::SDLViewer& viewer, NodeMapType& node_map, SpaceSearcherType& space_searcher) {
  // setup the bounding box for spacesearcher
  for (size_type i = 0; i < theta.size(); ++i) {
    Point bb_center(map_x_grid_range(theta[i][0]), map_y_grid_range(theta[i][1]), 0);
    BoundingBox bb(bb_center-.05, bb_center+.05);
    update(space_searcher.begin(bb), space_searcher.end(bb), 4*N+i, InTrianglePred(theta[i]));
  }
  #pragma omp parallel 
  {
    post_process(mesh);
  }

  auto max_node_it = std::max_element(mesh.node_begin(), mesh.node_end(), NodeValueLessComparator());
  double max_val = (*max_node_it).value();

  if (show_zaxis) {
    viewer.add_nodes(mesh.node_begin(), mesh.node_end(),HeatmapColor(max_val), node_map);
  } else {
    viewer.add_nodes(mesh.node_begin(), mesh.node_end(),
                   HeatmapColor(max_val), NodePosition(max_val), node_map);
  }

  if (show_label_state == 1) {
    viewer.set_label(N);
  } else if (show_label_state == 2) {
    viewer.set_label(accept_count / (value_type)(4*N) * 100.0);
  } else {
    viewer.set_label("");
  }
  
  CS207::sleep(sleep_interval);
}

/* Event handler of Graph trajectory display for MCMC_Iteration_Event */
void callback_trajectory(const std::vector<param_type>& theta, size_type i, size_type accept_count, 
  GraphType& graph, CS207::SDLViewer& viewer, NodeMapType& node_map) {
  (void) accept_count;
  
  // Make a trajectory per core, which each runs its own MCMC.
  size_type num_threads = omp_get_max_threads();
  static std::vector<std::vector<param_type>> trajectory_vec_list(num_threads);
  
  #pragma omp parallel 
  {
    size_type omp_id = omp_get_thread_num();
  //for (size_type omp_id = 0; omp_id != num_threads; ++omp_id) {
    param_type theta_omp_id = theta[omp_id];
    std::vector<param_type>& trajectory_vec = trajectory_vec_list[omp_id];
    
    // Force the tie conditional to be true if it is the first sample.
    auto last_position = Point(theta_omp_id[0]-1, theta_omp_id[1]-1, 0);
    if (i != 1) {
      last_position = Point(trajectory_vec.back()[0], trajectory_vec.back()[1], 0);
    }
    // Check tie condition; if it is true, new point was accepted and thus be added, 
    // else new point was rejected.
    if (std::tie(theta[omp_id][0], theta_omp_id[1]) != std::tie(last_position.x, last_position.y)) {
      if (trajectory_vec.size() >= TRAJECTORY_LENGTH) {
        // Pop out the first element after reaching the desired TRAJECTORY_LENGTH.
        trajectory_vec.erase(trajectory_vec.begin());
      }
      trajectory_vec.push_back(theta_omp_id);
    }
  
    // Update trajectory.
    size_type end = (trajectory_vec.size() == 1) ? TRAJECTORY_LENGTH : trajectory_vec.size();
    for (size_type k = 0; k < end; ++k) {
      size_type block_idx = omp_id * TRAJECTORY_LENGTH + k;
      // Force all node positions to have the same position if there is only
      // one point in the trajectory.
      size_type temp = (trajectory_vec.size() == 1) ? 0 : k;
      graph.node(block_idx).position() = Point(map_x_grid_range(trajectory_vec[temp][0]), map_y_grid_range(trajectory_vec[temp][1]), 1);
      graph.node(block_idx).value() = omp_id;
    }  
  }
  viewer.add_nodes(graph.node_begin(), graph.node_end(), TrajectoryColor(), node_map);
}

/* Callback function to handle keyboard events */
void callback_keyboard(SDLKey key) {
  switch(key) {
    case SDLK_DOWN:
      // Slow down the procedure.
      sleep_interval += sleep_step;
      sleep_interval = sleep_interval > .05 ? .05 : sleep_interval;
      std::cout << "Framerate:" << sleep_interval << std::endl;
      break;
    case SDLK_UP:
      // Speed up the procedure.
      sleep_interval -= sleep_step;
      sleep_interval = sleep_interval < 0. ? 0. : sleep_interval;
      std::cout << "Framerate:" << sleep_interval << std::endl;
      break;
    case SDLK_t:
      // Cycle between displaying num_iterations, acceptance_ratio, or nothing.
      ++show_label_state;
      show_label_state = show_label_state % 3;
      break;
    case SDLK_f:
      // Toggle display of the z-axis.
      show_zaxis = !show_zaxis;
      break;
    default:
      break;
  }
}

int main() {
  auto mesh = Grid()(std::make_pair(0., GRID_WIDTH), std::make_pair(0., GRID_HEIGHT), 51, 51);
  GraphType graph = Trajectory()(TRAJECTORY_LENGTH);
  // Print out statistics of the mesh.
  std::cout << mesh.num_nodes() << " "
            << mesh.num_edges() << " "
            << mesh.num_triangles() << std::endl;

  // Launch the SDLViewer.
  CS207::SDLViewer viewer;
  viewer.launch();
  
  auto node_map = viewer.empty_node_map(mesh);
  
  SpaceSearcherType space_searcher(mesh.triangle_begin(), mesh.triangle_end(), TriangleToPoint());

  viewer.add_nodes(mesh.node_begin(), mesh.node_end(), node_map);
  viewer.add_edges(mesh.edge_begin(), mesh.edge_end(), node_map);

  viewer.add_nodes(graph.node_begin(), graph.node_end(), node_map);
  viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);
  viewer.add_keyboard_listener(boost::bind(&callback_keyboard, _1));
  viewer.center_view();
  
  // Initialize MCMC simulator and problem set.
  MCMC_Simulator simulator;
  ProblemFactory::ProblemFactory problem_factory;
  
  simulator.add_mcmc_iteration_listener(boost::bind(&callback_distribution, _1, _2, _3, 
    boost::ref(mesh), boost::ref(viewer), boost::ref(node_map), boost::ref(space_searcher)));
  #if 1
  simulator.add_mcmc_iteration_listener(boost::bind(&callback_trajectory, _1, _2, _3, 
    boost::ref(graph), boost::ref(viewer), boost::ref(node_map)));
  #endif
  // Create problems definition from factory.
  
  auto data_range = problem_factory.create_range();
  set_range(data_range);
  // Simulate!
  simulator.simulate(problem_factory, 1e5, 2);
}