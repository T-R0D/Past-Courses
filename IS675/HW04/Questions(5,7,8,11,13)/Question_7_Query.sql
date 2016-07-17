-- 7. Summarize the amount of time worked in the TimeSheet table by job and task. 
-- Sort the result table by taskID within jobID. 
-- The result table should produce43 rows.

SELECT 
	JobID,
	TaskID,
	Count(*) 'NumberOfTimeCards',
	SUM(HoursWorked) 'TotalHoursWorked'
FROM 
	TimeSheet
GROUP BY
	JobID,TaskID
ORDER BY
	JobID,TaskID