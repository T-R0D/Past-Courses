/**
 * IS675 - HW06 - Team08
 * Using SQL Operations with More Complex Queries
 *
 * Division of Labor
 * Well...
 */

/**
 * Exercise 01
 *
 * Summarize the actual hours worked and labor cost (hours worked * hourly
 * pay rate) by JobID and TaskID for all rows in the JobTask table. Remember
 * that an employee's pay rate changes by date, so it is necessary to locate
 * the correct pay rate by date as you did for question #15 for HW#5. If you
 * were not able to get question #15 on HW#5 to work, I recommend doing that
 * before starting on this question. Compare the actual labor hours to the
 * estimated labor hours, and the actual labor costs to the estimated
 * labor costs.
 */
DROP VIEW [v_TimeSheetsWithPay];
CREATE VIEW [v_TimeSheetsWithPay] AS
SELECT
	[TimeSheet].[EmpID] AS 'EmpID',
	[Employee].[LastName] + ', ' + [Employee].[FirstName] AS 'Employee Name',
    [TimeSheet].[JobID] AS 'JobID',
	[TimeSheet].[TaskID] AS 'TaskID',
    [Job].[JobCompleted] AS 'JobCompleted',
	[TimeSheet].[StartWork] AS 'StartWork',
	[TimeSheet].[HoursWorked] AS 'HoursWorked',
	[EmployeePay].[HourlyPayRate] AS 'HourlyPay',
	[TimeSheet].[HoursWorked] * [EmployeePay].[HourlyPayRate] AS 'LaborCost'
FROM
	[TimeSheet]
		INNER JOIN [EmployeePay] ON
			[TimeSheet].[EmpID] = [EmployeePay].[EmpID]
		INNER JOIN [Employee] ON
			[TimeSheet].[EmpID] = [Employee].[EmpID]
        INNER JOIN [Job] ON
            [TimeSheet].[JobID] = [Job].[JobID]
WHERE
	[EmployeePay].[DateStartPay] <= [TimeSheet].[StartWork] AND
	[TimeSheet].[StartWork] < ISNULL([EmployeePay].[DateEnd], GETDATE()) AND
    [TimeSheet].[JobID] IS NOT NULL
;

DROP VIEW [v_ActualLaborByJobTask];
CREATE VIEW [v_ActualLaborByJobTask] AS
SELECT
    [v_TimeSheetsWithPay].[JobID] AS 'JobID',
    [v_TimeSheetsWithPay].[TaskID] AS 'TaskID',
    SUM([v_TimeSheetsWithPay].[HoursWorked]) AS 'HoursWorked',
    SUM(
        [v_TimeSheetsWithPay].[HoursWorked] * [v_TimeSheetsWithPay].[HourlyPay]
    ) AS 'LaborCost'
FROM
    [v_TimeSheetsWithPay]
GROUP BY
    [v_TimeSheetsWithPay].[JobID],
    [v_TimeSheetsWithPay].[TaskID]
;

DROP VIEW [v_LaborCostComparisons];
CREATE VIEW [v_LaborCostComparisons] AS
SELECT
    [JobTask].[JobID] AS 'JobID',
    [JobTask].[TaskID] AS 'TaskID',
    [Task].[TaskDescription] AS 'TaskDescription',
    CONVERT(varchar, [JobTask].[DateStarted], 101) AS 'DateStarted',
    ISNULL(
        CONVERT(varchar, [JobTask].[DateCompleted], 101),
        'Not Done'
    ) AS 'DateCompleted',
    [JobTask].[EstHours] AS 'EstHours',
    ISNULL([v_ActualLaborByJobTask].[HoursWorked], 0.00) AS 'ActualHoursWorked',
    [JobTask].[EstHours] - ISNULL([v_ActualLaborByJobTask].[HoursWorked], 0.00)
         AS 'LaborHoursVariance',
    [JobTask].[EstLaborCost] AS 'EstLaborCost',
    ISNULL([v_ActualLaborByJobTask].[LaborCost], 0.00) AS 'ActualLaborCost',
    [JobTask].[EstLaborCost] -
        ISNULL([v_ActualLaborByJobTask].[LaborCost], 0.00) AS 'LaborCostVariance'
FROM
    [JobTask]
        INNER JOIN [Task] ON
            [JobTask].[TaskID] = [Task].[TaskID]
        LEFT OUTER JOIN [v_ActualLaborByJobTask] ON
            [JobTask].[JobID] = [v_ActualLaborByJobTask].[JobID] AND
            [JobTask].[TaskID] = [v_ActualLaborByJobTask].[TaskID]
;

SELECT
    *
FROM
    [v_LaborCostComparisons]
ORDER BY
    [v_LaborCostComparisons].[JobID],
    [v_LaborCostComparisons].[TaskID]
;


/**
 * Exercise 02
 *
 * Summarize the actual material costs by jobID and task ID and compare
 * them to the estimated material cost for each row in the JobTask table.
 * This is very similar to what you did for question #11 in HW#5, so this
 * should be fairly easy if you got question #11 to work. 
 */
DROP VIEW [v_ActualMaterialsByTask];
CREATE VIEW [v_ActualMaterialsByTask] AS
SELECT
    [MaterialAssigned].[JobID] AS 'JobID',
    [MaterialAssigned].[TaskID] AS 'TaskID',
    SUM([MaterialAssigned].[Quantity] * [MaterialPurchased].[CostPerUOM]) AS 'MaterialCost'
FROM
    [MaterialAssigned]
        INNER JOIN [MaterialPurchased] ON
            [MaterialAssigned].[POID] = [MaterialPurchased].[POID]
GROUP BY
    [MaterialAssigned].[JobID],
    [MaterialAssigned].[TaskID]
;

DROP VIEW [v_MaterialCostComparisons];
CREATE VIEW [v_MaterialCostComparisons] AS
SELECT
    [JobTask].[JobID] AS 'JobID',
    [JobTask].[TaskID] AS 'TaskID',
    [Task].[TaskDescription] AS 'TaskDescription',
    CONVERT(varchar, [JobTask].[DateStarted], 101) AS 'DateStarted',
    ISNULL(
        CONVERT(varchar, [JobTask].[DateCompleted], 101),
        'Not Done'
    ) AS 'DateCompleted',
    [JobTask].[EstMaterialCost] AS 'EstMaterialCost',
    ISNULL([v_ActualMaterialsByTask].[MaterialCost], 0.00) AS 'ActualMaterialCost',
    (
        [JobTask].[EstMaterialCost] -
        ISNULL([v_ActualMaterialsByTask].[MaterialCost], 0.00)
    ) AS 'MaterialCostVariance'
FROM
    [JobTask]
        INNER JOIN [Task] ON
            [JobTask].[TaskID] = [Task].[TaskID]
        LEFT OUTER JOIN [v_ActualMaterialsByTask] ON
            [JobTask].[JobID] = [v_ActualMaterialsByTask].[JobID] AND
            [JobTask].[TaskID] = [v_ActualMaterialsByTask].[TaskID]
;

SELECT
    *
FROM
    [v_MaterialCostComparisons]
ORDER BY
    [v_MaterialCostComparisons].[JobID],
    [v_MaterialCostComparisons].[TaskID]
;


/**
 * Exercise 03
 *
 * Now it’s time to put them together. Compare actual to estimated costs
 * for each row in the JobTask table. The PercentVariance is the percentage
 * variance between the TotalEstCost and the TotalActualCost. The
 * general calculation is:
 * ((TotalEstCost – TotalActualCost)/TotalEstCost) * 100. 
 */

CREATE VIEW [v_TotalCostsByTask] AS
SELECT
    [v_LaborCostComparisons].[JobID] AS 'JobID',
    [v_LaborCostComparisons].[TaskID] AS 'TaskID',
    [v_LaborCostComparisons].[TaskDescription] AS 'TaskDescription',
    [v_LaborCostComparisons].[DateStarted] AS 'DateStarted',
    [v_LaborCostComparisons].[DateCompleted] AS 'DateCompleted',
    (
        [v_LaborCostComparisons].[EstLaborCost] +
        [v_MaterialCostComparisons].[EstMaterialCost]
    ) AS 'TotalEstCost',
    (
        [v_LaborCostComparisons].[ActualLaborCost] +
        [v_MaterialCostComparisons].[ActualMaterialCost]
    ) AS 'TotalActualCost'
FROM
    [v_LaborCostComparisons]
        INNER JOIN [v_MaterialCostComparisons] ON
            [v_LaborCostComparisons].[JobID] = [v_MaterialCostComparisons].[JobID] AND
            [v_LaborCostComparisons].[TaskID] = [v_MaterialCostComparisons].[TaskID]
;

DROP VIEW [v_AllCostComparisonsByTask];
CREATE VIEW [v_AllCostComparisonsByTask] AS
SELECT
    [v_TotalCostsByTask].[JobID] AS 'JobID',
    [v_TotalCostsByTask].[TaskID] AS 'TaskID',
    [v_TotalCostsByTask].[TaskDescription] AS 'Task Description',
    [v_TotalCostsByTask].[DateStarted] AS 'DateStarted',
    [v_TotalCostsByTask].[DateCompleted] AS 'DateCompleted',
    [v_LaborCostComparisons].[EstHours] AS 'EstHours',
    [v_LaborCostComparisons].[ActualHoursWorked] AS 'ActualHoursWorked',
    [v_LaborCostComparisons].[LaborHoursVariance] AS 'LaborHoursVariance',
    [v_LaborCostComparisons].[EstLaborCost] AS 'EstLaborCost',
    [v_LaborCostComparisons].[ActualLaborCost] AS 'ActualLaborCost',
    [v_LaborCostComparisons].[LaborCostVariance] AS 'LaborCostVariance',
    [v_MaterialCostComparisons].[EstMaterialCost] AS 'EstMaterialCost',
    [v_MaterialCostComparisons].[ActualMaterialCost] AS 'ActualMaterialCost',
    [v_MaterialCostComparisons].[MaterialCostVariance] AS 'MaterialCostVariance',
    [v_TotalCostsByTask].[TotalEstCost] AS 'TotalEstCost',
    [v_TotalCostsByTask].[TotalActualCost] AS 'TotalActualCost',
    (
        [v_TotalCostsByTask].[TotalEstCost] -
        [v_TotalCostsByTask].[TotalActualCost]
    ) AS 'TotalCostVariance',
    (
        (
            [v_TotalCostsByTask].[TotalEstCost] -
            [v_TotalCostsByTask].[TotalActualCost]
        ) *
        (
            100.00 /
            [v_TotalCostsByTask].[TotalEstCost]
        )
    ) AS 'PercentVariance'
FROM
    [v_TotalCostsByTask]
        INNER JOIN [v_LaborCostComparisons] ON
            [v_TotalCostsByTask].[JobID] = [v_LaborCostComparisons].[JobID] AND
            [v_TotalCostsByTask].[TaskID] = [v_LaborCostComparisons].[TaskID]
        INNER JOIN [v_MaterialCostComparisons] ON
            [v_TotalCostsByTask].[JobID] = [v_MaterialCostComparisons].[JobID] AND
            [v_TotalCostsByTask].[TaskID] = [v_MaterialCostComparisons].[TaskID]
;

SELECT
    *
FROM
    [v_AllCostComparisonsByTask]
ORDER BY
    [v_AllCostComparisonsByTask].[JobID],
    [v_AllCostComparisonsByTask].[TaskID]
;


/**
 * Exercise 04
 *
 * Summarize the information created in question #3 by job. The result
 * table should have one row per job in the Job table. Add additional data
 * from the Job and Client tables to provide more information about each
 * job in the result table. Hint: \textbf{\underline{Calculate}} the
 * PercentVariance – you cannot sum that field.
 */
DROP VIEW [v_TotalCostsByJob];
CREATE VIEW [v_TotalCostsByJob] AS
SELECT
    [v_AllCostComparisonsByTask].[JobID] AS 'JobID',
    SUM([v_AllCostComparisonsByTask].[EstHours]) AS 'EstHours',
    SUM([v_AllCostComparisonsByTask].[ActualHoursWorked]) AS 'ActualHoursWorked',
    SUM([v_AllCostComparisonsByTask].[LaborHoursVariance]) AS 'LaborHoursVariance',
    SUM([v_AllCostComparisonsByTask].[EstLaborCost]) AS 'EstLaborCost',
    SUM([v_AllCostComparisonsByTask].[ActualLaborCost]) AS 'ActualLaborCost',
    SUM([v_AllCostComparisonsByTask].[LaborCostVariance]) AS 'LaborCostVariance',
    SUM([v_AllCostComparisonsByTask].[EstMaterialCost]) AS 'EstMaterialCost',
    SUM([v_AllCostComparisonsByTask].[ActualMaterialCost]) AS 'ActualMaterialCost',
    SUM([v_AllCostComparisonsByTask].[MaterialCostVariance]) AS 'MaterialCostVariance',
    SUM([v_AllCostComparisonsByTask].[TotalEstCost]) AS 'TotalEstCost',
    SUM([v_AllCostComparisonsByTask].[TotalActualCost]) AS 'TotalActualCost',
    SUM([v_AllCostComparisonsByTask].[TotalCostVariance]) AS 'TotalCostVariance',
    (
        SUM([v_AllCostComparisonsByTask].[TotalCostVariance]) * 100 /
        SUM([v_AllCostComparisonsByTask].[TotalEstCost])
    ) AS 'PercentVariance'
FROM
    [v_AllCostComparisonsByTask]
GROUP BY
    [v_AllCostComparisonsByTask].[JobID]
;

DROP VIEW [v_JobInfo];
CREATE VIEW [v_JobInfo] AS 
SELECT
    [Job].[JobID] AS 'JobID',
    [Job].[JobName] AS 'JobName',
    [Client].[ClientName] AS 'ClientName',
    CASE
        WHEN [Job].[JobCompleted] = 1
            THEN 'Finished'
        ELSE
            'Not Finished'
    END /**/ AS 'JobStatus'
FROM
    [Job]
        INNER JOIN [Client] ON
            [Job].[ClientID] = [Client].[ClientID]
;

DROP VIEW [v_JobSummary];
CREATE VIEW [v_JobSummary] AS
SELECT
    [v_JobInfo].[JobID] AS 'JobID',
    [v_JobInfo].[JobName] AS 'JobName',
    [v_JobInfo].[ClientName] AS 'ClientName',
    [v_JobInfo].[JobStatus] AS 'JobStatus',
    [v_TotalCostsByJob].[EstHours] AS 'EstHours',
    [v_TotalCostsByJob].[ActualHoursWorked] AS 'ActualHoursWorked',
    [v_TotalCostsByJob].[LaborHoursVariance] AS 'LaborHoursVariance',
    [v_TotalCostsByJob].[EstLaborCost] AS 'EstLaborCost',
    [v_TotalCostsByJob].[ActualLaborCost] AS 'ActualLaborCost',
    [v_TotalCostsByJob].[LaborCostVariance] AS 'LaborCostVariance',
    [v_TotalCostsByJob].[EstMaterialCost] AS 'EstMaterialCost',
    [v_TotalCostsByJob].[ActualMaterialCost] AS 'ActualMaterialCost',
    [v_TotalCostsByJob].[MaterialCostVariance] AS 'MaterialCostVariance',
    [v_TotalCostsByJob].[TotalEstCost] AS 'TotalEstCost',
    [v_TotalCostsByJob].[TotalActualCost] AS 'TotalActualCost',
    [v_TotalCostsByJob].[TotalCostVariance] AS 'TotalCostVariance',
    [v_TotalCostsByJob].[PercentVariance] AS 'PercentVariance'
FROM
    [v_JobInfo]
        LEFT OUTER JOIN [v_TotalCostsByJob] ON
            [v_JobInfo].[JobID] = [v_TotalCostsByJob].[JobID]
;

SELECT
    *
FROM
    [v_JobSummary]
ORDER BY
    [v_JobSummary].[JobID]
;

/**
 * Exercise 05
 *
 * Which job that is \underline{finished} had actual total costs that were
 * closest to the estimated total costs? (PercentVariance closest to zero)
 * Make sure that the query could select the correct job from any data
 * set – the query should not just work with our test data set.
 */
SELECT
    *
FROM
    [v_JobSummary]
WHERE
    [v_JobSummary].[PercentVariance] = (
        SELECT
            MIN(ABS([v_JobSummary].[PercentVariance]))
        FROM
            [v_JobSummary]
    )
;


/**
 * Exercise 06
 *
 * Which job that is \underline{finished} had the largest percentage
 * positive labor hours variance? In other words, which finished job was
 * able to be completed with the least number of labor hours, when compared
 * to the estimated labor hours? The percentage labor hours variance is
 * calculated as the LaborHoursVariance/EstHours * 100. Add in the name of
 * the employee who served as the manager for the job.
 */
DROP VIEW [v_LaborHoursVariance];
CREATE VIEW [v_LaborHoursVariance] AS
SELECT
    [v_JobSummary].[JobID],
    [v_JobSummary].[JobName],
    [v_JobSummary].[ClientName],
    [v_JobSummary].[EstHours],
    [v_JobSummary].[ActualHoursWorked],
    [v_JobSummary].[EstHours] - [v_JobSummary].[ActualHoursWorked] AS 'LaborHoursVariance',
    ([v_JobSummary].[EstHours] - [v_JobSummary].[ActualHoursWorked]) * 100 / [v_JobSummary].[EstHours] AS 'PercentHoursVariance'
FROM
    [v_JobSummary]
WHERE
    [v_JobSummary].[JobStatus] = 'Finished'
;

DROP VIEW [v_JobManagerNames];
CREATE VIEW [v_JobManagerNames] AS
SELECT
    [Job].[JobID] AS 'JobID',
    [Employee].[LastName] + ', ' + [Employee].[FirstName] AS 'EmployeeManager'
FROM
    [Job]
        INNER JOIN [Employee] ON
            [Job].[EmpManagerID] = [Employee].[EmpID]
;
        

SELECT
    [v_LaborHoursVariance].[JobID],
    [v_LaborHoursVariance].[JobName],
    [v_LaborHoursVariance].[ClientName],
    [v_JobManagerNames].[EmployeeManager] AS 'EmployeeManager',
    [v_LaborHoursVariance].[EstHours],
    [v_LaborHoursVariance].[ActualHoursWorked],
    [v_LaborHoursVariance].[LaborHoursVariance],
    [v_LaborHoursVariance].[PercentHoursVariance]
FROM
    [v_LaborHoursVariance]
        INNER JOIN [v_JobManagerNames] ON
            [v_LaborHoursVariance].[JobID] = [v_JobManagerNames].[JobID]
WHERE
    [v_LaborHoursVariance].[PercentHoursVariance] =
    (SELECT
        MAX([v_LaborHoursVariance].[PercentHoursVariance])
      FROM
        [v_LaborHoursVariance]
    )
ORDER BY
    [v_LaborHoursVariance].[JobID]
;


/**
 * Exercise 07
 *
 * What is the average amount of time (labor hours) spent on a
 * \underline{completed} job task per square foot, as compared to the
 * estimated amount of time that should be spent on a task per square foot?
 *
 * Use the data in the JobTask table to calculate the average amount of
 * EstHours/Squarefeet, but use the data in the TimeSheet table to calculate
 * the average amount of time that was actually worked on a completed task.
 * I recommend creating separate views for the estimated hours per square
 * feet and the actual hours per square feet. The estimate view is a little
 * easier to create because it doesn’t require a join. Include all rows in the
 * JobTask table to get the average EstHours/Squarefeet for a task. To get the
 * average \underline{actual} hours per square feet requires that you join the
 * TimeSheet table and the JobTask table to be able to use the square feet in
 * the JobTask table. Do not include data for incompleted tasks when
 * calculating the ActualHours/SquareFeet. Remember that you have to SUM the
 * HoursWorked in the TimeSheet table by JobID and TaskID to get the Actual
 * HoursWorked from the TimeSheet table. I rounded the final results to 6
 * digits after the decimal point. The result table is at the top of the
 * next page. There is one row in the result table for each row in the Task
 * table. Sort the result table by TaskID.
 *
 * The ComparisonMessage should be generated as shown on the result table
 * above; if both the EstimatedHours and ActualHours are null, then put the
 * message ``Null Estimate" in the ComparisonMessage column. Remember that
 * a CASE statement in the SELECT list executes sequentially, so whatever
 * WHEN statement is placed first will be executed first. The CASE
 * statement stops executing as soon as a WHEN condition is true.
 * Potential problem: EstHours and Squarefeet are integers and must
 * be converted to decimal data types before they can be used in a
 * calculation that could generate a decimal result. 
 */
DROP VIEW  [v_EstHoursPerSqFt];
CREATE VIEW [v_EstHoursPerSqFt] AS
SELECT
    [JobTask].[TaskID] AS 'TaskID',
    SUM([JobTask].[EstHours]) /
        CAST(SUM([JobTask].[SquareFeet]) AS DECIMAL(6, 2)) AS 'EstimatedHoursPerSqFt'
FROM
    [JobTask]
GROUP BY
    [JobTask].[TaskID]
;

----------
CREATE VIEW v_CompleteSqFtSum AS
SELECT
    TaskID,
    SUM(SquareFeet) 'SquareFeet'
FROM
    JobTask
WHERE
    DateCompleted IS NOT NULL
GROUP BY
    TaskID
;

CREATE VIEW v_CompleteActualHours AS
SELECT
    TimeSheet.TaskID,
    SUM(TimeSheet.HoursWorked) 'Hours'
FROM
    TimeSheet
        INNER JOIN JobTask ON
            TimeSheet.JobID = JobTask.JobID AND TimeSheet.TaskID = JobTask.TaskID
WHERE
    JobTask.DateCompleted IS NOT NULL
GROUP BY
    TimeSheet.TaskID
;

CREATE VIEW v_CompleteActualHoursPerSqFt AS
SELECT
    A.TaskID,
    A.[Hours] / S.SquareFeet 'ActualHoursPerSqFt'
FROM
    v_CompleteActualHours A
        INNER JOIN v_CompleteSqFtSum S ON
            A.TaskID = S.TaskID
;

------------------------------
DROP VIEW [v_HoursPerSqFtComparison];
CREATE VIEW [v_HoursPerSqFtComparison] AS
SELECT
    [Task].[TaskID] AS 'TaskID',
    [Task].[TaskDescription] AS 'TaskDescription',
    [v_EstHoursPerSqFt].[EstimatedHoursPerSqFt] AS 'EstimatedHoursPerSqFt',
    [v_CompleteActualHoursPerSqFt].[ActualHoursPerSqFt] AS 'ActualHoursPerSqFt',
    CASE
        WHEN [v_EstHoursPerSqFt].[EstimatedHoursPerSqFt] IS NULL
            THEN 'Null Estimate'
        WHEN [v_CompleteActualHoursPerSqFt].[ActualHoursPerSqFt] IS NULL
            THEN 'Null Actual'
        WHEN [v_EstHoursPerSqFt].[EstimatedHoursPerSqFt] > [v_CompleteActualHoursPerSqFt].[ActualHoursPerSqFt]
            THEN 'Estimate Larger'
        WHEN [v_CompleteActualHoursPerSqFt].[ActualHoursPerSqFt] > [v_EstHoursPerSqFt].[EstimatedHoursPerSqFt]
            THEN 'Actual Larger'
        ELSE
            'No Difference'
    END /**/ AS 'Comparison Message'
FROM
    [Task]
        LEFT OUTER JOIN [v_EstHoursPerSqFt] ON
            [Task].[TaskID] = [v_EstHoursPerSqFt].[TaskID]
        LEFT OUTER JOIN [v_CompleteActualHoursPerSqFt] ON
            [Task].[TaskID] = [v_CompleteActualHoursPerSqFt].[TaskID]
;


/**
 * Exercise 08
 *
 * Use the result table generated for question #7 to help you answer
 * this question. The goal of this query is to identify which task has
 * the largest negative difference between the EstimatedHoursPerSqFt and
 * ActualHoursPerSqFt (which estimate is the worst because the actual
 * is larger).
 */
SELECT
    *
FROM
    v_HoursPerSqFtComparison Comparison
WHERE
    Comparison.EstimatedHoursPerSqFt - Comparison.ActualHoursPerSqFt = (
        SELECT
            MIN(EstimatedHoursPerSqFt - ActualHoursPerSqFt)
        FROM
            v_HoursPerSqFtComparison
    )
;


/**
 * Exercise 09
 *
 * The objective of this query is similar to that for question #7, except
 * this time we are going to look at labor costs rather than labor hours.
 * What is the average estimated labor cost per square foot as compared
 * to the actual labor cost per square foot for each task? I recommend
 * looking back at question #1, where you probably created a view to help
 * you calculate actual labor costs for a task on a job. That view will help
 * you with this question. Do \textbf{\underline{not}} include data for
 * incompleted tasks when calculating the actual labor cost/SquareFeet;
 * do include data for incompleted tasks when calculating the estimated
 * labor cost/squarefeet. 
 */
DROP VIEW  [v_EstWagesPerSqFt];
CREATE VIEW [v_EstWagesPerSqFt] AS
SELECT
    [JobTask].[TaskID] AS 'TaskID',
    CAST(SUM([JobTask].[EstLaborCost]) AS DECIMAL) /
        CAST(SUM([JobTask].[SquareFeet]) AS DECIMAL(6, 2)) AS 'EstimatedWagesPerSqFt'
FROM
    [JobTask]
GROUP BY
    [JobTask].[TaskID]
;

----------
CREATE VIEW v_ActualPay AS
SELECT
    TS.TaskID,
    SUM(TS.LaborCost) 'ActualWagesPerSqFt'
FROM
    v_TimeSheetsWithPay TS
        INNER JOIN JobTask JT ON
            TS.JobID = JT.JobID AND TS.TaskID = JT.TaskID
WHERE
    JT.DateCompleted IS NOT NULL
GROUP BY
    TS.TaskID
;

DROP VIEW [v_ActualWagesPerSqFt];
CREATE VIEW [v_ActualWagesPerSqFt] AS
SELECT
    [v_ActualPay].[TaskID] AS 'TaskID',
    v_ActualPay.ActualWagesPerSqFt /
        v_CompleteSqFtSum.SquareFeet AS 'ActualWagesPerSqFt'
FROM
    [v_ActualPay]
        INNER JOIN [v_CompleteSqFtSum] ON
            [v_ActualPay].[TaskID] = [v_CompleteSqFtSum].[TaskID]
;

------------------------------
SELECT
    [Task].[TaskID] AS 'TaskID',
    [Task].[TaskDescription] AS 'TaskDescription',
    [v_EstWagesPerSqFt].[EstimatedWagesPerSqFt] AS 'EstimatedLaborCostPerSqFt',
    [v_ActualWagesPerSqFt].[ActualWagesPerSqFt] AS 'ActualLaborCostPerSqFt',
    CASE
        WHEN [v_EstWagesPerSqFt].[EstimatedWagesPerSqFt] IS NULL
            THEN 'Null Estimate'
        WHEN [v_ActualWagesPerSqFt].[ActualWagesPerSqFt] IS NULL
            THEN 'Null Actual'
        WHEN [v_EstWagesPerSqFt].[EstimatedWagesPerSqFt] > [v_ActualWagesPerSqFt].[ActualWagesPerSqFt]
            THEN 'Estimate Larger'
        WHEN [v_ActualWagesPerSqFt].[ActualWagesPerSqFt] > [v_EstWagesPerSqFt].[EstimatedWagesPerSqFt]
            THEN 'Actual Larger'
        ELSE
            'No Difference'
    END /**/ AS 'Comparison Message'
FROM
    [Task]
        LEFT OUTER JOIN [v_EstWagesPerSqFt] ON
            [Task].[TaskID] = [v_EstWagesPerSqFt].[TaskID]
        LEFT OUTER JOIN [v_ActualWagesPerSqFt] ON
            [Task].[TaskID] = [v_ActualWagesPerSqFt].[TaskID]
;


/**
 * Exercise 10
 *
 * Which clients did not have any jobs with a DateAccepted last year?
 * Which materials were not assigned (DateAssigned) to any job tasks last year?
 * Combine the clients and materials into a single result table (hint: Use the
 * UNION statement). Make sure that you use the GETDATE() function to
 * determine the correct year.
 */
DROP VIEW [v_ClientsWithoutJobsInPreviousYear];
CREATE VIEW [v_ClientsWithoutJobsInPreviousYear] AS
SELECT
    [Client].[ClientID] AS 'ClientID',
    [Client].[ClientName] AS 'ClientName'
FROM
    [Client]
WHERE
    [Client].[ClientID] NOT IN (
        SELECT
            ClientID 
        FROM
               Job
        WHERE
            DATEDIFF(YEAR, DateAccepted, GETDATE()) = 1
    )
;

DROP VIEW v_MaterialsAssignedYearPrevious;
CREATE VIEW v_MaterialsAssignedYearPrevious AS
SELECT
    DISTINCT(Material.MaterialID)
FROM
    MaterialAssigned
        INNER JOIN MaterialPurchased ON
            MaterialAssigned.POID = MaterialPurchased.POID
        INNER JOIN Material ON
            MaterialPurchased.MaterialID = Material.MaterialID
WHERE
    DATEDIFF(YEAR, MaterialAssigned.DateAssigned, GETDATE()) = 1
;

DROP VIEW v_MaterialsNotAssignedYearPrevious;
CREATE VIEW v_MaterialsNotAssignedYearPrevious AS
SELECT
    MaterialID,
    MaterialName
FROM
    Material
WHERE
    MAterial.MaterialID NOT IN (SELECT * FROM v_MaterialsAssignedYearPrevious)


SELECT
    'Client: ' + CAST(ClientID AS VARCHAR) AS 'ClientOrMaterialID',
    'Client: ' + ClientName AS 'ClientOrMaterialName'
FROM
    v_ClientsWithoutJobsInPreviousYear
/**/
UNION
/**/
SELECT
    'Material: ' + CAST(MaterialID AS VARCHAR) AS 'ClientOrMaterialID',
    'Material: ' + MaterialName AS 'ClientOrMaterialName'
FROM
    v_MaterialsNotAssignedYearPrevious
; 


/**
 * Drop any created views here to reduce DBO clutter
 */
DROP VIEW [v_TimeSheetsWithPay];
DROP VIEW [v_ActualLaborByJobTask];
DROP VIEW [v_LaborCostComparisons];
DROP VIEW [v_ActualMaterialsByTask];
DROP VIEW [v_MaterialCostComparisons];
DROP VIEW [v_TotalCostsByTask];
DROP VIEW [v_AllCostComparisonsByTask];
DROP VIEW [v_TotalCostsByJob];
DROP VIEW [v_JobInfo];
DROP VIEW [v_JobSummary];
DROP VIEW [v_LaborHoursVariance];
DROP VIEW [v_JobManagerNames];



DROP VIEW [v_ClientsWithoutJobsInPreviousYear];