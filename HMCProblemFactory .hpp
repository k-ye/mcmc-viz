#include "HMCModel.hpp"
#include "vector"

struct HMCProblemFactory {
// define your model types into the type names used by visualizer
#if 1
	typedef HMCModel::ProblemProposal ProblemProposal;
	typedef HMCModel::ProblemRange ProblemRange;
	typedef HMCModel::ProblemInitialParams ProblemInitialParams;
#endif

	typedef unsigned size_type;
	typedef double value_type;
	typedef std::vector<value_type> param_type;

	/* Create proposal object of the model. */
	ProblemProposal create_proposal() const {
		return {};
	}

	/* Create parameter range of the model. */
	ProblemRange create_range() const {
		return {};
	}

	/* Create initial parameters for simulation. */
	std::vector<std::vector<param_type>> create_initial_params() const {
		auto initial_params_object = ProblemInitialParams();
		return initial_params_object.initial_params();
	}
};