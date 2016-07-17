--Display information from the JobTask table about job tasks that were completed two years ago. List the JobID, TaskID, DateStarted, the estimated material
--cost per square foot, the estimated labor cost per square foot, and the estimated labor cost per estimated hour. Sort the result table by taskID within jobid.
--Make sure that your query determine which tasks were completed two years ago based on the current year. For example, we all know that the current year is
--2015, so two years ago is 2013. However, I don’t want you to code ‘2013’ in the SQL WHERE clause. I want this query to be able to be run in 2016, and
--have it show tasks from two years ago as 2014 without having to change the actual query code.

SELECT JobID,
	   TaskID,
	   DateStarted,
	   EstMaterialCost/SquareFeet 'EstMaterialCostperSqft',
	   EstLaborCost/SquareFeet 'EstLaborCostperSqft',
	   EstLaborCost/EstHours 'EstLaborCostperHr'
FROM JobTask
WHERE DATEDIFF(YYYY,  DateCompleted, GETDATE()) = 2
ORDER BY TaskID;