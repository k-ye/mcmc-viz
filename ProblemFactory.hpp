#include "Waterbuck.hpp"


typedef Waterbuck::ProblemProposal ProblemProposal;
typedef Waterbuck::ProblemPosterior ProblemPosterior;
typedef Waterbuck::ProblemRange ProblemRange;

struct ProblemFactory {
	ProblemProposal create_proposal() {
		return {};
	}

	ProblemPosterior create_posterior() {
		return {};
	}

	ProblemRange create_range() {
		return {};
	}
};