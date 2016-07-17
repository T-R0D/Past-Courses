--> Question 1, Can be Stored Procedure.. Paramemter: IssueReportId
SELECT 
	a_IssueReport.IssueReportId,
	a_IssueReport.ProductSerialNumber,
	a_IssueReport.ReportDate,
	a_TestForm.TestCompleted,
	DATEDIFF(DAY,a_IssueReport.ReportDate,a_TestForm.TestCompleted) 'Days in System',
	a_Unit.ModelNumber,
	CASE 
		WHEN a_IssueReport.CustomerId is not null
			THEN a_customer.FirstName +' ' + a_Customer.LastName
		WHEN a_IssueReport.EmployeeId is not null
			THEN a_Employee.FirstName +' ' + a_Employee.LastName
		WHEN a_IssueReport.DistributorId is not null
			THEN a_Distributor.Name
	END 'Reporter',
	CASE 
		WHEN a_IssueReport.CustomerId is not null
			THEN 'Customer'
		WHEN a_IssueReport.EmployeeId is not null
			THEN 'Employee'
		WHEN a_IssueReport.DistributorId is not null
			THEN 'Distributor'
	END 'Reporter Type',
	a_IssueReportProblem.Description 'Type of Problem',
	a_IssueReport.ProblemDescription 'Description of Problem',
	CASE
		WHEN a_IssueReport.InjuryOccurred = 1
			THEN 'Yes'
		ELSE
			'No'
	END 'Injury',
	CASE
		WHEN a_IssueReport.InjuryOccurred = 1
			THEN a_IssueReport.InjuryDescription
		ELSE
			'None'
	END 'Injury'
FROM 
	a_IssueReport
INNER JOIN 
	a_TestForm ON a_IssueReport.IssueReportId = a_TestForm.IssueReportId
INNER JOIN
	a_Unit ON a_IssueReport.ProductSerialNumber = a_Unit.SerialNumber 
LEFT JOIN
	a_Customer ON a_IssueReport.CustomerId = a_Customer.CustomerId
LEFT JOIN
	a_Employee ON a_IssueReport.EmployeeId = a_Employee.EmpId
LEFT JOIN
	a_Distributor ON a_IssueReport.DistributorId = a_Distributor.DistributorId
INNER JOIN 
	a_IssueReportProblem ON a_IssueReportProblem.ProblemID = a_IssueReport.IssueReportProblemId
WHERE 
	a_IssueReport.IssueReportId = 5
;
GO

-->  Question 2
SELECT 
	a_IssueReport.IssueReportId,
	a_IssueReport.ProductSerialNumber,
	a_IssueReport.ReportDate,
	a_TestForm.TestCompleted,
	a_Employee.FirstName + ' ' + a_Employee.LastName 'Tester',
	a_TestForm.TestDescription,
	a_TestForm.TestResults,
	CASE
		WHEN a_TestForm.TestCompleted is NULL
			THEN 'NO'
		ELSE
			'YES'
	END 'Test Completed?'
FROM 
	a_IssueReport
INNER JOIN 
	a_TestForm ON a_IssueReport.IssueReportId = a_TestForm.IssueReportId
INNER JOIN
	a_Unit ON a_IssueReport.ProductSerialNumber = a_Unit.SerialNumber
LEFT JOIN
	a_Employee ON a_TestForm.EmpId= a_Employee.EmpId 
WHERE 
	a_IssueReport.IssueReportId = 5
;
GO

/**
 * Number 03

 */
 /*
 CREATE VIEW av_ReturnedUnits --  the view was important to merge the data from model and unit.  Because 
 unit contained the fk to th model enitity which held the model description. We need to merge the two so that when we 
 needed to access the model description from enitities that only had the serial number we were able to do so via the unit entity
 or I should say from this view.
*/
CREATE VIEW av_ReturnedUnits AS
SELECT
    SerialNumber,
    a_Unit.ModelNumber,
    a_Model.[Description],
    MSRP
FROM
    a_Unit
        INNER JOIN a_Model ON
            a_Unit.ModelNumber = a_Model.ModelNumber
;
GO

 /*
 CREATE VIEW av_TestCounts was usefull because it allowed us to count the number of tests completed for each issue report.
*/
CREATE VIEW av_TestCounts AS
SELECT
    Test.IssueReportId,
    COUNT(*) AS 'TestsConducted'
FROM
    a_TestForm Test
GROUP BY
    Test.IssueReportId
;
GO

 /*
 CREATE VIEW av_Number3 was important because it utilized the two prior views and merged their data with the 
 additional data we needed to complete the queury for problem number 3.  By using mulitple view for this problem, we were
 able to modularize a large query into smaller more manageable queries.  We were also able to extract data from this view
 for query number 4 with a where condition.
*/

CREATE VIEW av_Number3 AS
SELECT
    Report.IssueReportId AS 'Report ID',
    Report.ReportDate AS 'Date Made',
    Report.ProductSerialNumber AS 'Product Serial Number',
    Unit.ModelNumber AS 'Model #',
    Unit.[Description] AS 'Model Description',
    Problem.[Description] AS 'ProblemType',
    ISNULL(TestCounts.TestsConducted, 0) AS 'Number of Tests Performed',
    CASE
        WHEN Distributor.DistributorId IS NOT NULL
            THEN Distributor.Name
        WHEN Employee.EmpId IS NOT NULL
            THEN Employee.LastName + ', ' + Employee.FirstName 
        WHEN Customer.CustomerId IS NOT NULL
            THEN Customer.LastName + ', ' + Customer.FirstName
    END AS 'Reporter Name',
    CASE
        WHEN Distributor.DistributorId IS NOT NULL
            THEN 'Distributor'
        WHEN Employee.EmpId IS NOT NULL
            THEN 'Employee'
        WHEN Customer.CustomerId IS NOT NULL
            THEN 'Customer'
    END AS 'ReporterType'
FROM
    a_IssueReport Report
        INNER JOIN a_IssueReportProblem Problem ON
            Report.IssueReportProblemId = Problem.[ProblemID]
        INNER JOIN av_ReturnedUnits Unit ON
            Report.ProductSerialNumber = Unit.SerialNumber
        LEFT OUTER JOIN av_TestCounts TestCounts ON
            Report.IssueReportId = TestCounts.IssueReportId
        LEFT OUTER JOIN a_Distributor Distributor ON
            Report.DistributorId = Distributor.DistributorId
        LEFT OUTER JOIN a_Customer Customer ON
            Report.CustomerId = Customer.CustomerId
        LEFT OUTER JOIN a_Employee Employee ON
            Report.EmployeeId = Employee.EmpId
;
GO

SELECT
    *
FROM
    av_Number3
;
GO

/**
 * Number 04
 */
SELECT
    *
FROM
    av_Number3
WHERE
    av_Number3.[Number of Tests Performed] = 0
;
GO

/**
 * Number 05
 */

/*
 CREATE VIEW av_Number5 wasn't really important for us to create for query number five, however looking ahead to query number six, we knew we had 
 to extract information from within this query for a particular condition so we decided to store the query as a view.  This enabled
 us to utilize the query again without having to copy and pasting the same code. 
*/
CREATE VIEW av_Number5 AS
SELECT
    Report.IssueReportId AS 'Report ID',
    Report.ReportDate AS 'Date Made',
    Report.ProductSerialNumber AS 'Product Serial Number',
    Unit.ModelNumber AS 'Model #',
    Unit.[Description] AS 'Model Description',
    Problem.[Description] AS 'ProblemType',
    ISNULL(TestCounts.TestsConducted, 0) AS 'Number of Tests Performed',
    CASE
        WHEN Distributor.DistributorId IS NOT NULL
            THEN Distributor.Name
        WHEN Employee.EmpId IS NOT NULL
            THEN Employee.LastName + ', ' + Employee.FirstName 
        WHEN Customer.CustomerId IS NOT NULL
            THEN Customer.LastName + ', ' + Customer.FirstName
    END AS 'Reporter Name',
    CASE
        WHEN Distributor.DistributorId IS NOT NULL
            THEN 'Distributor'
        WHEN Employee.EmpId IS NOT NULL
            THEN 'Employee'
        WHEN Customer.CustomerId IS NOT NULL
            THEN 'Customer'
    END AS 'ReporterType'
FROM
    a_IssueReport Report
        INNER JOIN a_IssueReportProblem Problem ON
            Report.IssueReportProblemId = Problem.[ProblemID]
        INNER JOIN av_ReturnedUnits Unit ON
            Report.ProductSerialNumber = Unit.SerialNumber
        LEFT OUTER JOIN av_TestCounts TestCounts ON
            Report.IssueReportId = TestCounts.IssueReportId
        LEFT OUTER JOIN a_Distributor Distributor ON
            Report.DistributorId = Distributor.DistributorId
        LEFT OUTER JOIN a_Customer Customer ON
            Report.CustomerId = Customer.CustomerId
        LEFT OUTER JOIN a_Employee Employee ON
            Report.EmployeeId = Employee.EmpId
WHERE
    Report.IssueReportId IN (
        SELECT
            TF.IssueReportId
        FROM
            a_TestForm TF
        WHERE
            TF.TestCompleted IS NULL
    )
;
GO

SELECT
    *
FROM
    av_Number5
;
GO

/**
 * Number 6
 */
SELECT
    *
FROM
    av_Number5 OuterNumber5
WHERE
    OuterNumber5.[Date Made] = (
        SELECT
            MIN(InnerNumber5.[Date Made])
        FROM
            av_Number5 InnerNumber5
    )
;
GO

/**
 * Question #7
 */

/*
 CREATE VIEW  v_ModelReportCount,  v_ModelTestCount, v_ModelInjuryCount, and v_ModelReportDate were useful because they allowed us
 to summerize the data for each model number and description.  By breaking up the query into multiple views we were able to gather
 specific data (such as count of probs,tests,problem reports) for each vehicle model.  This allowed for more logical approach to the ultimate
 query and we were able to debug any errors we had in our enity attributes as we went.  The views also allowed us to easily use aggregate functions 
 without the complexity of having to keep track of all the joins or groups.
*/

CREATE VIEW  v_ModelReportCount AS 
SELECT
	a_Unit.ModelNumber,
	Count(*) 'Report Count'
FROM
	a_IssueReport
LEFT JOIN a_Unit
	ON a_Unit.SerialNumber = a_IssueReport.ProductSerialNumber
GROUP BY a_Unit.ModelNumber
;
GO

CREATE VIEW v_ModelTestCount AS
SELECT
	a_Unit.ModelNumber,
	Count(*) 'Test Count'
FROM
	a_TestForm
LEFT JOIN a_IssueReport
	ON a_IssueReport.IssueReportId = a_TestForm.IssueReportId
LEFT JOIN a_Unit
	ON a_IssueReport.ProductSerialNumber = a_Unit.SerialNumber
Group BY a_Unit.ModelNumber
;
GO

CREATE VIEW v_ModelInjuryCount AS
SELECT
	a_Unit.ModelNumber,
	COUNT(*) 'Injury Count'
FROM 
	a_IssueReport
LEFT JOIN a_Unit
	on a_IssueReport.ProductSerialNumber = a_Unit.SerialNumber
WHERE a_IssueReport.InjuryOccurred = 1
Group BY a_Unit.ModelNumber 
;
GO

CREATE VIEW v_ModelReportDate AS
SELECT
	a_IssueReport.ProductSerialNumber,
	a_Unit.ModelNumber,
	a_issueReport.ReportDate
FROM 
	a_IssueReport
LEFT JOIN a_Unit
	on a_IssueReport.ProductSerialNumber = a_Unit.SerialNumber
;
GO

/*
 CREATE VIEW   v_ModelSummary, this view allowed us to bring all the data from the four prior views into one 
 qeury and summarize all the data for each model into one place.  We also knew we needed to use this data for query number eight
 so it made sense to create a view that we could reference down the road.  
*/

CREATE VIEW v_ModelSummary AS
SELECT
	a_Model.ModelNumber,
	a_Model.[Description],
	ISNULL(v_ModelReportCount.[Report Count],0) 'Count of Problem Reports',
	ISNULL(v_ModelTestCount.[Test Count],0) 'Count of Tests',
	ISNULL(v_ModelInjuryCount.[Injury Count],0) 'Count of Injury Problem Reports',
	(SELECT MIN(v_ModelReportDate.ReportDate) FROM v_ModelReportDate WHERE a_Model.ModelNumber = v_ModelReportDate.ModelNumber) 'Earliest Problem Report Date',
	(SELECT MAX(v_ModelReportDate.ReportDate) FROM v_ModelReportDate WHERE a_Model.ModelNumber = v_ModelReportDate.ModelNumber) 'Most Recent Problem Report Date'
FROM 
	a_Model
LEFT JOIN v_ModelReportCount
	ON a_Model.ModelNumber = v_ModelReportCount.ModelNumber
LEFT JOIN v_ModelTestCount
	ON a_Model.ModelNumber = v_ModelTestCount.ModelNumber
LEFT JOIN v_ModelInjuryCount
	ON a_Model.ModelNumber = v_ModelInjuryCount.ModelNumber
GROUP BY
	a_Model.ModelNumber,a_Model.[Description],v_ModelReportCount.[Report Count],v_ModelTestCount.[Test Count],v_ModelInjuryCount.[Injury Count]
;
GO


/**
 * Question 8
 */

 /*
 CREATE VIEW   v_ModelMinReport and  v_ModelMaxReport allowed us to complete this query systematically.  We originally tried to 
 write the query for number eight without any views and it got really messy, so we opted to use the modular approach once again and do
 views so we could see what the results as we went.  Due to the nature of query number eight, we concluded that doing views to determine 
 which vehilce had the minimum amount of issues (as in v_ModelminReport) and the maximimum amount of issues (as in v_ModelMaxReport)
 was appropriate because each of these views required a sub query to compute the min and max report vehicles. These views were helpful
 because for the completion of query number eight we just had to create a select statment that took the values produced from the two
 views and ORed (unioned) the results to complete the final query.
*/

CREATE VIEW v_ModelMinReport AS
SELECT
	v_ModelSummary.ModelNumber,
	v_ModelSummary.Description,
	v_ModelSummary.[Count of Problem Reports]
FROM
	v_ModelSummary
WHERE
	v_ModelSummary.[Count of Problem Reports]  = (SELECT MIN(v_ModelSummary.[Count of Problem Reports]) FROM v_ModelSummary)
;
GO

CREATE VIEW v_ModelMaxReport AS
SELECT
	v_ModelSummary.ModelNumber,
	v_ModelSummary.Description,
	v_ModelSummary.[Count of Problem Reports]
FROM
	v_ModelSummary
WHERE
	v_ModelSummary.[Count of Problem Reports]  = (SELECT MAX(v_ModelSummary.[Count of Problem Reports]) FROM v_ModelSummary)
;
GO

SELECT 
	*,
	'Min' AS MinOrMax
FROM
	v_ModelMinReport
UNION
SELECT 
	*,
	'Max' AS MinOrMax
FROM
	v_ModelMaxReport
;
GO

/**
 * Number 9
 */

  /*
 CREATE VIEW   av_ReporterTypeFrequency was useful because it allowed us to count the number of issue reports made by 
 each entity (customer, distributor, employee).  From this we could create a subquery that just pulled the entity that 
 filed the most reports.  This query was useful in keeping the whole query simple.  Like most other views we have used in 
 this project, the purpose was to be able to see our logic and program in modular fashion.
*/
CREATE VIEW av_ReporterTypeFrequency AS
SELECT
    av_Number3.ReporterType,
    COUNT(*) AS 'Frequency'
FROM
    av_Number3
GROUP BY
    av_Number3.ReporterType
;
GO

SELECT
    RTF.ReporterType,
    RTF.Frequency
FROM
    av_ReporterTypeFrequency RTF
WHERE
    RTF.Frequency = (
        SELECT
            MAX(InnerRTF.Frequency)
        FROM
            av_ReporterTypeFrequency InnerRTF
    )
;
GO
