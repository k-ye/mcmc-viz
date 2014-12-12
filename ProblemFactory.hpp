#include "Waterbuck.hpp"
#include "SimpleProblem.hpp"
#include "vector"

namespace ProblemFactory {
// define your model types into the type names used by visualizer
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

	/* Problem factory to create definition set of the model. */
	struct ProblemFactory {
		/* Create proposal object of the model. */
		ProblemProposal create_proposal() const {
			return {};
		}

		/* Create posterior objet of the model. */
		ProblemPosterior create_posterior() const {
			return {};
		}

		/* Create parameter range of the model. */
		ProblemRange create_range() const {
			return {};
		}

		/* Create initial parameters for simulation. */
		std::vector<param_type> create_initial_params() const {
			auto initial_params_object = ProblemInitialParams();
			return initial_params_object.initial_params();
		}
	};
}