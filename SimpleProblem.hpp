#include <fstream>
#include <cmath>
#include <random>
#include <vector>

#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/normal_distribution.hpp>

namespace SimpleProblem {
	class SimpleProblemProposal;
	class SimpleProblemPosterior;
	struct SimpleProblemRange;
	struct SimpleProblemInitialParams;

	typedef SimpleProblemPosterior ProblemPosterior;
	typedef SimpleProblemProposal ProblemProposal;
	typedef SimpleProblemRange ProblemRange;
	typedef SimpleProblemInitialParams ProblemInitialParams;

	typedef double value_type;
	typedef unsigned size_type;
	typedef std::vector<value_type> param_type;

	const double PI = boost::math::constants::pi<value_type>();

	/* Generate bivariate (independent) normal samples. */
	class SimpleProblemProposal {
	 public:
	  
	  SimpleProblemProposal(value_type mu_x, value_type mu_y, value_type sigma_x, value_type sigma_y)
	    : mu_x_(mu_x), mu_y_(mu_y), sigma_x_(sigma_x), sigma_y_(sigma_y) {
	  }

	  SimpleProblemProposal() : mu_x_(1.), mu_y_(1.), sigma_x_(.5), sigma_y_(.5) {

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

	class SimpleProblemPosterior {
	 public:
	  SimpleProblemPosterior(value_type mu_x, value_type mu_y, value_type sigma_x, value_type sigma_y)
	    : mu_x_(mu_x), mu_y_(mu_y), sigma_x_(sigma_x), sigma_y_(sigma_y) {
	  }

	  SimpleProblemPosterior() : mu_x_(1.), mu_y_(1.), sigma_x_(0.1), sigma_y_(0.1) {

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

	struct SimpleProblemRange {
	  std::pair<value_type, value_type> x;
	  std::pair<value_type, value_type> y;

	  SimpleProblemRange() : x(std::make_pair(0., 2.)), y(std::make_pair(0., 2.)) {

	  }
	};

	struct SimpleProblemInitialParams {
		param_type initial_params() {
			return {1., 1.};
		}
	};
}