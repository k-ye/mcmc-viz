#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/normal_distribution.hpp>

namespace HMCModel {
	class HMCModelProposal;
	class HMCModelPosterior;
	class HMCModelRange;
	class HMCModelInitialParames;

	typedef HMCModelProposal ProblemProposal;
	typedef HMCModelPosterior ProblemPosterior;
	typedef HMCModelRange ProblemRange;
	typedef HMCModelInitialParames ProblemInitialParams;

	typedef double value_type;
	typedef unsigned size_type;
	typedef std::vector<value_type> param_type;

	class HMCModelPosterior {
	public:
		// p(x,y) = 1 - x^2 - y^2
		value_type operator()(const param_type& theta_, bool log_) const {
			double den = 1 - pow(theta_[0], 2) - pow(theta_[1], 2);
			return log_ ? log(den) : den;
		}

	private:
		friend class HMCModelProposal;
	};

	class HMCModelProposal {
		// Engine type to generate random numbers
		typedef boost::mt19937 ENG;
		// Distribution type to be used by pdf() in boost.
		typedef boost::math::normal_distribution<value_type> normal_dens_type;
		// Distribution type to generate random numbers.
		typedef boost::normal_distribution<value_type> normal_rand_type;
		// Random number generator type.
		typedef boost::variate_generator<ENG, normal_rand_type> RAND_GEN;
	public:
		HMCModelProposal() : delta_(.02), L_(40),
		momt_generator_(RAND_GEN(ENG(), normal_rand_type(0., 2.))) {
			
		}

		param_type rand(const param_type& pos, size_type) {
			// start from previous position
			param_type proposed_pos = pos;
			// random sample a new momentum to perturb
			param_type proposed_momt(pos.size());
			proposed_momt[0] = std::max(-1., std::min(1., momt_generator_()));
			proposed_momt[1] = std::max(-1., std::min(1., momt_generator_()));

			for (size_type i = 0; i < L_; ++i) {
				leap_frog(proposed_pos, proposed_momt);
			}

			return proposed_pos;
		}

	private:
		param_type U_derivative(const param_type& pos) const {
			param_type deriv(2);
			// dU/dx = -2x
			deriv[0] = -2 * pos[0];
			// dU/dy = -2y
			deriv[1] = -2 * pos[1];
			return deriv;
		}

		param_type K_derivative(const param_type& momt) const {
			param_type deriv(2);
			// dK/dp_x = p_x / sigma_x^2
			deriv[0] = momt[0] / 2.;
			// dK/dp_y = p_y / sigma_y^2
			deriv[1] = momt[1] / 2.;
			return deriv;
		}

		// Update @a pos and @a momt
		void leap_frog(param_type& pos, param_type& momt) const {
			// 1. THe first 0.5 delta leap for momentum:
			auto U_deriv = U_derivative(pos);
			for (size_type i = 0; i < momt.size(); ++i) {
				momt[i] -= 0.5 * delta_ * U_deriv[i];
			}
			// 2. 1 delta leap for pos:
			auto K_deriv = K_derivative(momt);
			for (size_type i = 0; i < pos.size(); ++i) {
				pos[i] += delta_ * K_deriv[i];
			}
			// 3. The rest 0.5 delta leap for momentum:
			U_deriv = U_derivative(pos);
			for (size_type i = 0; i < momt.size(); ++i) {
				momt[i] -= 0.5 * delta_ * U_deriv[i];
			}

			std::cout << "pos_x: " << pos[0] << ", pos_y: " << pos[1] << std::endl;
			std::cout << "momt_x: " << momt[0] << ", momt_y: " << momt[1] << std::endl;
			std::cout << std::endl;
		}

		// time step
		double delta_;
		// times to do leap frog
	  size_type L_;
	  RAND_GEN momt_generator_;
	};
}