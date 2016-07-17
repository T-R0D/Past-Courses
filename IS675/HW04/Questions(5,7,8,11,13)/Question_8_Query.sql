-- Modify the query written for question #7 to display only those rows where the NumberofTimeCards is greater than 5.
-- For any timecards with null values forthe jobID or taskID, 
-- put a message in the JobID and TaskID columns. Sort the result table by TaskID within JobID. 

SELECT 
	ISNULL(CONVERT(VARCHAR, JobID, 120),'No JobID') 'JobID',
	ISNULL(CONVERT(VARCHAR, TaskID, 120),'No TaskID') 'TaskID',
	COUNT(*)  'NumberOfTimeCards',
	SUM(HoursWorked) 'TotalHoursWorked'
FROM 
	TimeSheet
GROUP BY
	JobID,TaskID
HAVING 
	COUNT(*) > 5
ORDER BY
	JobID ,TaskID

		