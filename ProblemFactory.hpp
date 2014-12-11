#include "Waterbuck.hpp"
//#include "SimpleProblem.hpp"
#include "vector"

namespace ProblemFactory {
#if 1
	typedef Waterbuck::ProblemProposal ProblemProposal;
	typedef Waterbuck::ProblemPosterior ProblemPosterior;
	typedef Waterbuck::ProblemRange ProblemRange;
	typedef Waterbuck::ProblemInitialParams ProblemInitialParams;
#endif
#if 0
	typedef SimpleProblem::ProblemProposal ProblemProposal;
	typedef SimpleProblem::ProblemPosterior ProblemPosterior;
	typedef SimpleProblem::ProblemRange ProblemRange;
	typedef SimpleProblem::ProblemInitialParams ProblemInitialParams;
#endif
	typedef unsigned size_type;
	typedef double value_type;
	typedef std::vector<value_type> param_type;

	struct ProblemFactory {
		ProblemProposal create_proposal() const {
			return {};
		}

		ProblemPosterior create_posterior() const {
			return {};
		}

		ProblemRange create_range() const {
			return {};
		}

		std::vector<param_type> create_initial_params() const {
			auto initial_params_object = ProblemInitialParams();
			return initial_params_object.initial_params();
		}
	};
}