#include <vector>
#include <boost/signals2/signal.hpp>
#include <boost/function.hpp>
#include <omp.h>
#include "HMCProblemFactory.hpp"

typedef HMCProblemFactory::size_type size_type;
typedef HMCProblemFactory::value_type value_type;
typedef HMCProblemFactory::param_type param_type;
// The proposal type of the stats model
typedef HMCProblemFactory::ProblemProposal ProblemProposal;
typedef HMCProblemFactory::ProblemRange ProblemRange;

/* Class to run MCMC simulation given a specific problem set. */
class HMC_Simulator {
 public:
  // EventHandler type for MCMC iteration event.
  typedef boost::function<void (const std::vector<param_type>&, size_type, size_type)> HMC_Iteration_Event;

  /** Run @a max_N times iteration of the MCMC sampling procedure. Paralleled for multiple core computers.
    * @param[in] problem_factory: ProblemFactory object that defines the stats model.
    * @param[in] max_N: number of iteration to run.
    * @param[in] p: number of dimension.
    *
    * Time Complexity: O(@a max_N)
    */
  void simulate(const HMCProblemFactory& problem_factory, size_type max_N, size_type p) {
    size_type num_threads = omp_get_max_threads();
    // Initialize matrix of size max_N x p.
    std::vector<param_type>theta((max_N+1)*num_threads);
    std::vector<param_type> momt((max_N+1)*num_threads);
    // create problem definition from problem factory
    ProblemProposal proposal = problem_factory.create_proposal();
    auto initial_val = problem_factory.create_initial_params();
    assert(initial_val[0].size() >= num_threads);
    
    for (size_type i = 0; i < num_threads; ++i) {
      theta[i*max_N] = initial_val[0][i];
      momt[i*max_N] = initial_val[1][i];
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
        auto pos_momt_star = proposal.rand(theta[j-1], p);
        // new position and momentum
        auto pos_star = pos_momt_star[0];
        auto momt_star = pos_momt_star[1];
        // calculate the acceptance ratio
        value_type accept_ratio = proposal.H_energy(pos_star, momt_star, true) 
                                - proposal.H_energy(theta[j-1], momt[j-1], true);
        accept_ratio = std::min(1., exp(accept_ratio));
        // decide to accept or reject proposed sample point
        if (runif(generator) < accept_ratio) {
          theta[j] = pos_star;
          accept_bit[omp_id] = 1;
        }
        else {
          theta[j] = theta[j-1];
          accept_bit[omp_id] = 0;
        }
        momt[j] = momt_star;
        proposed_theta[omp_id] = theta[j];
      }
      for (auto bit : accept_bit) accept_count += bit;
      // fire the event to tell the handler that this iteration has completed
      signals_(proposed_theta, i, accept_count);      
    }
  }

  /* Connect an event listener to @a signals_ for MCMC_Iteration_Event registration. */
  void add_mcmc_iteration_listener(const HMC_Iteration_Event& func) {
    signals_.connect(func);
  }

private:
  /* Event listener containers */
  boost::signals2::signal<void (const std::vector<param_type>&, size_type, size_type)> signals_;
};
