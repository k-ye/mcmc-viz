#include <vector>
#include <boost/signals2/signal.hpp>
#include <boost/function.hpp>
#include <omp.h>
#include "ProblemFactory.hpp"

typedef ProblemFactory::size_type size_type;
typedef ProblemFactory::value_type value_type;
typedef ProblemFactory::param_type param_type;
// The proposal type of the stats model
typedef ProblemFactory::ProblemProposal ProblemProposal;
// The posterior type of the stats model
typedef ProblemFactory::ProblemPosterior ProblemPosterior;
typedef ProblemFactory::ProblemRange ProblemRange;

/* Class to run MCMC simulation given a specific problem set. */
class MCMC_Simulator {
 public:
  // EventHandler type for MCMC iteration event.
  typedef boost::function<void (const std::vector<param_type>&, size_type, size_type)> MCMC_Iteration_Event;

  /** Run @a max_N times iteration of the MCMC sampling procedure. Paralleled for multiple core computers.
    * @param[in] problem_factory: ProblemFactory object that defines the stats model.
    * @param[in] max_N: number of iteration to run.
    * @param[in] p: number of dimension.
    *
    * Time Complexity: O(@a max_N)
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
    
    for (size_type i = 1; i <= max_N; ++i) {
      // This is necessary because in parallel, we canot access directly to accept_count
      std::vector<size_type> accept_bit(num_threads, 0);
      // Store @num_threads proposed parameters
      std::vector<param_type> proposed_theta(num_threads);
      #pragma omp parallel 
      {
        size_type omp_id = omp_get_thread_num();
        size_type j = omp_id * max_N + i;
        // theta[j-1] is the previous state, theta_star is the new(proposed) state
        param_type theta_star = proposal.rand(theta[j-1], p);
        // calculate the acceptance ratio
        value_type accept_ratio = posterior(theta_star, true) - posterior(theta[j-1], true)
                                + proposal.dens(theta[j-1], theta_star, true) - proposal.dens(theta_star, theta[j-1], true);
        accept_ratio = std::min(1., exp(accept_ratio));
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

  /* Connect an event listener to @a signals_ for MCMC_Iteration_Event registration. */
  void add_mcmc_iteration_listener(const MCMC_Iteration_Event& func) {
    signals_.connect(func);
  }

private:
  /* Event listener containers */
  boost::signals2::signal<void (const std::vector<param_type>&, size_type, size_type)> signals_;
};
