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
