#ifndef __PROBLEMSTATE_CPP__
#define __PROBLEMSTATE_CPP__

#include "ProblemState.hpp"


ProblemState::ProblemState() : mState{{3, 3, 1}} {
}

ProblemState::ProblemState(const ProblemState& pThat) {
	*this = pThat;
}

ProblemState& ProblemState::operator=(const ProblemState& pThat) {
	if (this != &pThat) {
		mState = pThat.mState;
		mActions = pThat.mActions;
	}

	return *this;
}

ProblemState ProblemState::applyAction(const std::vector<int>& pAction) {
	std::vector<int> action = pAction;

	if (mState[2] == 1) {
		for (int i = 0; i < 3; i++) {
			action[i] *= -1;
		}
	}

	ProblemState newState(*this);

	for (int i = 0; i < 3; i++) {
		newState.mState[i] += action[i];
	}
	newState.mActions.push_back(actionToString(pAction));

	return newState;
}

bool ProblemState::isValid() const {
	return validVector(mState) && validVector(complementVector(mState));
}

bool ProblemState::validVector(const std::vector<int>& pState) const {
	return (
		(pState[0] == 0 || (pState[0] >= pState[1])) &&
		pState[0] >= 0 &&
		pState[1] >= 0 &&
		pState[0] <= 3 &&
		pState[1] <= 3
	);
}

bool ProblemState::isSuccess() const {
	bool success = true;

	for (auto count : mState) {
		if (count != 0) {
			success = false;
		}
	}

	return success;
}

std::string ProblemState::toString() const {
	std::string ret;

	for (auto count : mState) {
		ret += ((char) count + '0');
	}

	return ret;
}

std::vector<int> ProblemState::complementVector(const std::vector<int>& pVector) const {
	std::vector<int> complement;

	for (auto count : pVector) {
		complement.push_back(3 - count);
	}
	complement[2] = 1 - pVector[2];

	return complement;
}

std::string ProblemState::actionToString(const std::vector<int>& pAction) const {
	std::string ret("");
	ret += (char(pAction[0]) + '0');
	ret += " missionaries ";
	ret += (char(pAction[1]) + '0');
	ret += " cannibals";

	return ret;
}

#endif //__PROBLEMSTATE_CPP__