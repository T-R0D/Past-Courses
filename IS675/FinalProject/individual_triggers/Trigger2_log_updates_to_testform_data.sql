CREATE TRIGGER trgAfterUpdate ON [dbo].[a_TestForm] 
FOR UPDATE
AS
	declare @empname varchar(100);
	declare @audit_action varchar(100);

	select @empname=(select user);	

	
	if update(TestFormId)
		set @audit_action='Updated TestFormId';
	if update(RecommendingTestFormId)
		set @audit_action='Updated RecommendingTestFormID.';
	if update(IssueReportId)
		set @audit_action='Updated IssueReportId.';
	if update(EmpId)
		set @audit_action='Updated EmpId.';
	if update(TestType)
		set @audit_action='Updated TestType.';
	if update(TestDescription)
		set @audit_action='Updated TestDescription.';
	if update(TestResults)
		set @audit_action='Updated TestResults.';
	if update(RecommendedResolution)
		set @audit_action='Updated RecommendedResolution.';
	if update(TestCompleted)
		set @audit_action='Update TestCompleted.';
	insert into  Employee_a_TestForm_Audit(Emp_Name,Audit_Action,Audit_Timestamp) 
	values(USER,@audit_action,getdate());

	PRINT 'Logged Update of Testform Table.'
GO

