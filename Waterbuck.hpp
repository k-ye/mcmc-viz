#include <fstream>
#include <cmath>
#include <random>
#include <vector>

#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/normal_distribution.hpp>

namespace Waterbuck {
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

	class WaterbuckPosterior {
		typedef boost::math::binomial_distribution<value_type> binomial;
	 public:
		WaterbuckPosterior() : data_({53., 57., 66., 67., 72.}) {

		}

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

	class WaterbuckProposal {
		typedef boost::mt19937 ENG;
		typedef boost::math::normal_distribution<value_type> normal_dens_type;
		typedef boost::normal_distribution<value_type> normal_rand_type;
		typedef boost::variate_generator<ENG, normal_rand_type> RAND_GEN;
	 public:
		WaterbuckProposal() : N_generator_(RAND_GEN(ENG(), normal_rand_type(0., sqrt(5.)))), theta_generator_(RAND_GEN(ENG(), normal_rand_type(0., sqrt(0.01)))) {
	  }

	  param_type rand(const param_type& theta, size_type) {
	    auto theta_sample = std::max(std::min(theta[1] + theta_generator_(), 1.), 0.);
	    auto N_sample = std::max(std::round(theta[0] + N_generator_()), 72.);
	    return {N_sample, theta_sample};
	  }
	  
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
	  std::pair<value_type, value_type> x;
	  std::pair<value_type, value_type> y;

	  WaterbuckRange() : x(std::make_pair(70., 250.)), y(std::make_pair(0., 1.)) {

	  }
	};

	struct WaterbuckInitialParams {
		param_type initial_params() {
			return {100., .5};
		}
	};
}