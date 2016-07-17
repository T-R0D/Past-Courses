--Let’s get familiar with the EmployeePay table in this query. 
--This query is a little difficult to understand, but very important to the next two homework assignments. 
--The EmployeePay table keeps track of the employee HourlyPayRate over time. The assumption is that an employee’s HourlyPayRate may change based on date. 
--In the EmployeePay table you created for HW#3, the HourlyPayRate always goes up over time. However, that may not always be true. 
--It is also possible that an HourlyPayRate might go down, depending on the economy, an employee’s performance, and an employee’s work circumstances (i.e. moving from full-time to part-time). 
--The ultimate goal for our job costing application is to make sure that the HourlyPayRate earned by an employee is correctly applied to the direct labor costs recorded on a time sheet.
--Since we aren’t joining tables in this assignment, I just want to be sure that you know how to access the pay rate for a particular employee for a particular date.
--The same general query should work for all of the questions below, you should only have to change the date in the WHERE clause to make them work. 
--You only have to turn in the SQL code for the first date, as long as that query will work for all 3.
SELECT EmpID,
       HourlyPayRate
FROM EmployeePay
WHERE EmpID = 6460 AND( ((convert(date, 'June 18, 2013')) 
BETWEEN DateStartPay AND DateEnd) OR ((convert (date, 'June 18, 2013') > DateStartPay AND DateEnd IS NULL)));
