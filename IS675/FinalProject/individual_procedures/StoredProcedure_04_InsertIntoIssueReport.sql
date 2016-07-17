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

EXECUTE  stpr_InsertIntoIssueReport 
	@ProductSerialNumber = 1234652,
	@DistributorId  =  NULL,
	@EmployeeId  = 1105,
	@CustomerId = NULL,
	@IssueReportProblemId = 2,
	@ProblemDescription = 'What''s the matter with this thing, what''s all that churnning and bubbling, you call that radar screen?',
	@ReportDate = 'July 4, 1987',
	@ReportMethodId = 5,
    @InjuryOccurred = 0,
    @InjuryDescription = 'None'
GO

SELECT * FROM a_IssueReport;