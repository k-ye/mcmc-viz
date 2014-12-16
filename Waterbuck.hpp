#include <fstream>
#include <cmath>
#include <random>
#include <vector>

#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/normal_distribution.hpp>

namespace Waterbuck {
	/** This model is using Waterbuck dataset: [53., 57., 66., 67., 72.]
		* The data is assumed to be sampled from a Binomial distribution (N, theta), where
		* a non-informative prior that is proportional to 1 / N has been used.
		*
		* Reference: Raftery, A. Inference for the binomial N parameter: a hierarchical Bayes approach. Biometrika 75, 223-8. 1988.
		*/

	class WaterbuckPosterior;
	class WaterbuckProposal;
	struct WaterbuckRange;
	struct WaterbuckInitialParams;

	typedef WaterbuckPosterior ProblemPosterior;
	typedef WaterbuckProposal ProblemProposal;
	typedef WaterbuckRange ProblemRange;
	typedef WaterbuckInitialParams ProblemInitialParams;

	typedef double value_type;
	typedef unsigned size_type;
	typedef std::vector<value_type> param_type;

	/* Class to calculate the posterior distribution of Waterbuck model. */
	class WaterbuckPosterior {
		typedef boost::math::binomial_distribution<value_type> binomial;
	 public:
		WaterbuckPosterior() : data_({53., 57., 66., 67., 72.}) {

		}

		/** Function to calculate the posterior density given the current parameter under the statistical model.
      * @param[in] param: current parameter
      * @param[in] log_: specify if returning density or log density
      * @result: density of @a theta_ under the model, @result when @a log_ == exp(@result) when !@a log_
      *
      * Time Complexity: O(1), since the length of the dataset is fixed.
      */
		value_type operator()(const param_type& theta_, bool log_) const {
			value_type N = theta_[0];
			value_type theta = theta_[1];
			value_type den = log(1. / N);
			binomial bin_distr_(N, theta);
			for (size_type i = 0; i < data_.size(); ++i) {
				den += log(pdf(bin_distr_, data_[i]));
			}
			return log_ ? den : exp(den);
		}

	private:
		param_type data_;
	};

	/** Class to wrap the propose distribution and its density. In this case we are using two independent normal distributions
		*	for N and theta. This works well if we are running MCMC in parallel with different initial position.
		*/
	class WaterbuckProposal {
		// Engine type to generate random numbers
		typedef boost::mt19937 ENG;
		// Distribution type to be used by pdf() in boost.
		typedef boost::math::normal_distribution<value_type> normal_dens_type;
		// Distribution type to generate random numbers.
		typedef boost::normal_distribution<value_type> normal_rand_type;
		// Random number generator type.
		typedef boost::variate_generator<ENG, normal_rand_type> RAND_GEN;
	 public:
		WaterbuckProposal() : N_generator_(RAND_GEN(ENG(), normal_rand_type(0., sqrt(5.)))), 
		theta_generator_(RAND_GEN(ENG(), normal_rand_type(0., sqrt(0.01)))) {
			
		}

	  /** Function to propose a random parameter that is of dimension @a prev_param.size().
      * @param[in] prev_param: the previous parameter in the MCMC chain
      * @param[in] iteration: current iteration
      * @result: new proposed paramter
      */
	  param_type rand(const param_type& theta, size_type) {
	    auto theta_sample = std::max(std::min(theta[1] + theta_generator_(), 1.), 0.);
	    auto N_sample = std::max(std::round(theta[0] + N_generator_()), 72.);
	    return {N_sample, theta_sample};
	  }
	  
	  /** Function to calculate the probability density of the state transition from @a param_from to @a param_to. 
	  	* This is useful in determining the acceptance ratio for the new proposed parameter.
      * @param[in] param_from, param_to: previous and new state of the parameter
      * @param[in] log_: specify if returning density or log density
      * @result: density of the transition, @result when @a log_ == exp(@result) when !@a log_
      */
	  value_type dens(const param_type& theta_to, const param_type& theta_from, bool log_) const {
	    normal_dens_type dist_N_ = normal_dens_type(0., sqrt(10.));
	    normal_dens_type dist_theta_ = normal_dens_type(0., sqrt(0.01));
	      
	    value_type N_to = theta_to[0];
	    value_type theta_to_ = theta_to[1];
	    value_type N_from = theta_from[0];
	    value_type theta_from_ = theta_from[1];
	    
	    value_type dens = 0;
	    dens += log(pdf(dist_N_, N_to - N_from));
	    dens += log(pdf(dist_theta_, theta_to_ - theta_from_));

	    return log_ ? dens : exp(dens);
	  }
	  
	 private:
	  RAND_GEN N_generator_;
	  RAND_GEN theta_generator_;
	};

	struct WaterbuckRange {
		// Range of parameter N, [@a x.first, @a x.second]
	  std::pair<value_type, value_type> x;
	  // Range of parameter theta, [@a x.first, @a x.second]
	  std::pair<value_type, value_type> y;

	  WaterbuckRange() : x(std::make_pair(70., 400.)), y(std::make_pair(0., 1.)) {

	  }
	};

	struct WaterbuckInitialParams {
		/* Make some guess of the initial parameters */
		std::vector<param_type> initial_params() {
			return {{100., .5}, {150., .4}, {200., .3}, {250., .15}};
		}
	};
}