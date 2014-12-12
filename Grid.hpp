#include "mesh/Mesh.hpp"

typedef unsigned size_type;
typedef double value_type;
typedef std::vector<value_type> param_type;
typedef Mesh<value_type, bool, value_type> MeshType;
// Interval type
typedef std::pair<value_type, value_type> interval;

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