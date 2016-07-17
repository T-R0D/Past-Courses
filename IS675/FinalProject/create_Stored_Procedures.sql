-- ================================================
-- Template generated from Template Explorer using:
-- Create Procedure (New Menu).SQL
--
-- Use the Specify Values for Template Parameters 
-- command (Ctrl-Shift-M) to fill in the parameter 
-- values below.
--
-- This block of comments will not be included in
-- the definition of the procedure.
-- ================================================
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Terence Henriod
-- Create date: 
-- Description:	
-- =============================================
CREATE PROCEDURE stpr_FindIssueReport
	@ReportId int = NULL
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
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
	    a_IssueReport.IssueReportId = @ReportId;
END
GO

-- ================================================
-- Template generated from Template Explorer using:
-- Create Procedure (New Menu).SQL
--
-- Use the Specify Values for Template Parameters 
-- command (Ctrl-Shift-M) to fill in the parameter 
-- values below.
--
-- This block of comments will not be included in
-- the definition of the procedure.
-- ================================================
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE stpr_IncompleteReports 
	-- Add the parameters for the stored procedure here
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
	SELECT *
		FROM av_Number3
		WHERE av_Number3.[Number of Tests Performed] = 0;
END
GO



-- ================================================
-- Template generated from Template Explorer using:
-- Create Procedure (New Menu).SQL
--
-- Use the Specify Values for Template Parameters 
-- command (Ctrl-Shift-M) to fill in the parameter 
-- values below.
--
-- This block of comments will not be included in
-- the definition of the procedure.
-- ================================================
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE stpr_oldest_open_report
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
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
END
GO



-- ================================================
-- Template generated from Template Explorer using:
-- Create Procedure (New Menu).SQL
--
-- Use the Specify Values for Template Parameters 
-- command (Ctrl-Shift-M) to fill in the parameter 
-- values below.
--
-- This block of comments will not be included in
-- the definition of the procedure.
-- ================================================
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		<Author,,Name>
-- Create date: <Create Date,,>
-- Description:	<Description,,>
-- =============================================
CREATE PROCEDURE stpr_InsertIntoIssueReport
	-- Add the parameters for the stored procedure here
	@ProductSerialNumber int,
	@DistributorId int = NULL,
	@EmployeeId int = NULL,
	@CustomerId int = NULL,
	@IssueReportProblemId int,
	@ProblemDescription varchar(1000) = NULL,
	@ReportDate date,
	@ReportMethodId int,
    @InjuryOccurred bit,
    @InjuryDescription varchar(1000)
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
	Insert into a_IssueReport
		(
			ProductSerialNumber,
			DistributorId ,
			EmployeeId ,
			CustomerId ,
			IssueReportProblemId ,
			ProblemDescription ,
			ReportDate  ,
			ReportMethodId  ,
			InjuryOccurred  ,
			InjuryDescription

		)
	Values
		(
			@ProductSerialNumber,
			@DistributorId,
			@EmployeeId,
			@CustomerId,
			@IssueReportProblemId,
			@ProblemDescription,
			@ReportDate,
			@ReportMethodId,
			@InjuryOccurred,
			@InjuryDescription
		)
END
GO




-- ================================================
-- Template generated from Template Explorer using:
-- Create Procedure (New Menu).SQL
--
-- Use the Specify Values for Template Parameters 
-- command (Ctrl-Shift-M) to fill in the parameter 
-- values below.
--
-- This block of comments will not be included in
-- the definition of the procedure.
-- ================================================
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE PROCEDURE stpr_MarkTestAsComplete 
	-- Add the parameters for the stored procedure here
	@TestFormId int = NULL,
	@TestOutcome varchar(1000) = "Outcome uncertain",
    @Recommendation varchar(1000) = "None"
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	-- SET NOCOUNT ON;

    -- Insert statements for procedure here
	UPDATE
        a_TestForm
    SET
        a_TestForm.TestResults = @TestOutcome,
        a_TestForm.RecommendedResolution = @Recommendation,
        a_TestForm.TestCompleted = GETDATE()
    WHERE
        a_TestForm.TestFormId = @TestFormId AND a_TestForm.TestCompleted IS NULL
END
GO


