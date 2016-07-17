SELECT 
		[Employee].EmpID AS 'EmpID',
		[Employee].LastName + ', '
		+ [Employee].FirstName  AS 'EmployeeName',
		--sum([timeSheet].HoursWorked) as 'TotalHoursWorked', 
		COUNT ([TimeSheet].TaskID) AS 'NumberofTimeSheets'
FROM   [TimeSheet]
JOIN   [Employee]
ON	   [TimeSheet].EmpID = [Employee].EmpID
GROUP BY [TimeSheet].EmpID
--ORDER BY [Employee].EmpID


--select * from employee;