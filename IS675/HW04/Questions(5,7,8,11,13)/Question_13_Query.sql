-- 13. Which client(s) do not currently have a Job in the job table? 
-- Hint: this query requires the use of a non-correlated sub-query. 

SELECT
	ClientName
FROM
	Client
WHERE 
	ClientID not in (
		SELECT 
			ClientID 
		FROM 
			JOB
	)
ORDER BY
	ClientName;
		 
