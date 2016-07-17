// hacky for now, to be cleaned later

#include <vector>
#include <string>
#include <cstdio>
#include <stack>
#include <set>
#include <utility>

#include "ProblemState.hpp"
#include "ProcessResult.cpp"


#define b printf("At line: %i", __LINE__);


ProcessResult<ProblemState> iterativeDeepeningSearch(
	const ProblemState& pInitialState,
	const int pAbsoluteMaxSearchDepth) {

	ProcessResult<ProblemState> result;
	bool searchEnded = false;
	int searchDepth = 1;
	int deepeningIncrement = 1;

	while (true) {
		searchDepth += deepeningIncrement;

		if (searchDepth >= pAbsoluteMaxSearchDepth) {
			result.hasFailed();
			return result;
		}

		std::stack< std::pair<ProblemState, int> > searchPath;
		searchPath.push({pInitialState, 0});

		while (!searchPath.empty()) {
			ProblemState currentState = searchPath.top().first;
			int currentDepth = searchPath.top().second;
			searchPath.pop();

			if (currentState.isSuccess()) {
				result.setResult(currentState);
				return result;
			}

			if (currentDepth + 1 <= searchDepth) {
				for (auto action : PROBLEM_ACTIONS) {
					ProblemState nextState = currentState.applyAction(action);

					if (nextState.isValid()) {
						searchPath.push({nextState, currentDepth + 1});
					}
				}
			}
		}
	}

	return result;
}



int main(int argc, char**argv) {
	ProblemState initialState;

	ProcessResult<ProblemState> problemSolution = iterativeDeepeningSearch(initialState, 20);

	if (problemSolution.isSuccess()) {
		puts("Solution:");
		for (auto action : problemSolution.getResult().mActions) {
			puts(action.c_str());
		}
	} else {
		puts("A solution could not be found...");
	}

	return 0;
}