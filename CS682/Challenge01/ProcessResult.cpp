#ifndef __PROCESSRESULT_CPP__
#define __PROCESSRESULT_CPP__

#include "ProcessResult.hpp"

template <typename ResultItem>
ProcessResult<ResultItem>::ProcessResult() {
	mProcessSuccess = false;
}


template <typename ResultItem>
void ProcessResult<ResultItem>::hasFailed() {
	mProcessSuccess = false;
}


template <typename ResultItem>
void ProcessResult<ResultItem>::setResult(const ResultItem pResult) {
	mProcessSuccess = true;
	mItem = pResult;
}


template <typename ResultItem>
bool ProcessResult<ResultItem>::isSuccess() const {
	return mProcessSuccess;
}


template <typename ResultItem>
ResultItem ProcessResult<ResultItem>::getResult() const {
	if (!isSuccess()) {
		throw std::exception();
	}

	return mItem;
}

#endif //__PROCESSRESULT_CPP__