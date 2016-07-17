// hacky for now, to be cleaned later

#include <vector>
#include <string>
#include <cstdio>
#include <set>
#include <queue>

std::vector< std::vector<int> > gActions = {
	{2, 0, 1},
	{1, 0, 1},
	{1, 1, 1},
	{0, 1, 1},
	{0, 2, 1}
};

class ProblemState {
 public:
	ProblemState() : mState{{3, 3, 1}} {
	}

	ProblemState(const ProblemState& pThat) {
		*this = pThat;
	}

	ProblemState& operator=(const ProblemState& pThat) {
		if (this != &pThat) {
			mState = pThat.mState;
			mActions = pThat.mActions;
		}

		return *this;
	}

	ProblemState applyAction(const std::vector<int>& pAction) {
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

	bool isValid() const {
		return validVector(mState) && validVector(complementVector(mState));
	}

	bool validVector(const std::vector<int>& pState) const {
		return (
			(pState[0] == 0 || (pState[0] >= pState[1])) &&
			pState[0] >= 0 &&
			pState[1] >= 0 &&
			pState[0] <= 3 &&
			pState[1] <= 3
		);
	}

	bool isSuccess() const {
		bool success = true;

		for (auto count : mState) {
			if (count != 0) {
				success = false;
			}
		}

		return success;
	}

	std::string toString() const {
		std::string ret;

		for (auto count : mState) {
			ret += ((char) count + '0');
		}

		return ret;
	}

	std::vector<int> complementVector(const std::vector<int>& pVector) const {
		std::vector<int> complement;

		for (auto count : pVector) {
			complement.push_back(3 - count);
		}
		complement[2] = 1 - pVector[2];

		return complement;
	}

	std::string actionToString(const std::vector<int>& pAction) const {
		std::string ret("");
		ret += (char(pAction[0]) + '0');
		ret += " missionaries ";
		ret += (char(pAction[1]) + '0');
		ret += " cannibals";

		return ret;
	}

	std::vector<int> mState;
	std::vector<std::string> mActions;
};

ProblemState bfs(const ProblemState& pInitialState) {
	ProblemState currentState;
	std::set<std::string> encounteredStates;
	std::queue<ProblemState> statesToBeProcessed;

	encounteredStates.insert(pInitialState.toString());
	statesToBeProcessed.push(pInitialState);

	while (!statesToBeProcessed.empty()) {
		currentState = statesToBeProcessed.front();
		statesToBeProcessed.pop();

		if (currentState.isSuccess()) {
			return currentState;
		} else {
			for (auto operation : gActions) {
				ProblemState newState = currentState.applyAction(operation);

				if (newState.isValid()) {
					if (encounteredStates.find(newState.toString()) == std::end(encounteredStates)) {
						encounteredStates.insert(newState.toString());
						statesToBeProcessed.push(newState);
					}
				}
			}
		}
	}

	throw 666;
}


int main(int argc, char**argv) {
	ProblemState initialState;

	try {
		ProblemState solution = bfs(initialState);
			puts("solution");
			int i;
			for (i = 0; i < solution.mActions.size() - 1; i += 2) {
				printf("Move %s across\n", solution.mActions[i].c_str());
				printf("Move %s back\n", solution.mActions[i + 1].c_str());
			}
			printf("Move %s across\n", solution.mActions[i].c_str());
	} catch (int e) {
		puts("no solution...");
	}

	return 0;
}