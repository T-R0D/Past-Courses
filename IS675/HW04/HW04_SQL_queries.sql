--problem 1

--problem 2

--problem 3
SELECT
	  JobID	'Job ID'
	, JobName	'Job Name'
	, DateProposed	'Date Proposed'
	, ISNULL(CONVERT(VARCHAR, DateAccepted, 120), 'Not Accepted')	'Date Accepted'
	, EmpManagerID	'Employee Manager ID'
	, PrimaryJobID	'Primary Job ID'
	, CASE
		WHEN JobCompleted = 1
			THEN 'Job Finished'
		ELSE
			'Job Not Finished'
	  END 'Job Completed Message'
FROM
	Job
ORDER BY
	DateProposed DESC
;

--problem 4

--problem 5

--problem 6
/** Assumption: if a job is not currently complete, it makes
 * no sense to compute a completion time yet - let the result
 * be NULL. Also, don't consider incomplete jobs.
 */
SELECT
	  JobID 'Job ID'
	, TaskID 'TaskID'
	, DateStarted 'Date Started'
	, DateCompleted 'Date Completed'
	, DATEDIFF(DAY, DateStarted, DateCompleted)
	, CASE
		WHEN DATEDIFF(DAY, DateStarted, DateCompleted) > 5
			THEN 'Late Completion - Investigate'
		ELSE
			'' -- No message for typical completion times was specified
	  END 'Message'
FROM
	JobTask
WHERE
	    DateCompleted IS NOT NULL
	AND	DATEDIFF(DAY, DateStarted, DateCompleted) > 3
ORDER BY
	  JobID
	, TaskID
;


--problem 7

--problem 8

--problem 9
SELECT
	  EmpID 'Employee ID'
	, SUM(HoursWorked) 'Total Hours Not Assigned To Job'
	, COUNT(*)'Time Sheets Not Assigned To Job'
FROM
	TimeSheet
WHERE
	Activity IS NOT NULL
GROUP BY
	EmpID
;


--problem 10

--problem 11

--problem 12
SELECT
	  YEAR(DateAssigned) 'Assigned Year'
	, JobID 'JobID'
	, SUM(Quantity) 'Total Material Quantity Assigned'
FROM
	MaterialAssigned
GROUP BY
	  YEAR(DateAssigned)
	, JobID
ORDER BY
	  YEAR(DateAssigned)
	, JobID
;

--problem 13

--problem 14

--problem 15
SELECT
	(LastName + ', ' + SUBSTRING(FirstName, 1, 1) + '.') 'Employee Name'
FROM
	Employee
WHERE
	EmpID NOT IN (
		SELECT
			EmpManagerID
		FROM
			Job
		WHERE
			EmpManagerID IS NOT NULL
	)
ORDER BY
	LastName
;