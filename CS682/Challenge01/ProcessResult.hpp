#ifndef __PROCESSRESULT_HPP__
#define __PROCESSRESULT_HPP__

#include <exception>

template <typename ResultItem>
class ProcessResult {
 public:
	ProcessResult();

	void hasFailed();
	void setResult(const ResultItem pResult);

	bool isSuccess() const;
	ResultItem getResult() const;

	bool mProcessSuccess;
	ResultItem mItem;
};

#endif //__PROCESSRESULT_HPP__