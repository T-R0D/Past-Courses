--Which client(s) had a job in the Job table where the year of the dateaccepted for the job was in 2014? 
--Hint: this query requires the use of a non-correlated sub-query. Result table:

SELECT ClientName,
	   ClientZip,
	   Email
From Client
WHERE ClientID In (
		SELECT ClientID
		FROM Job
		WHERE 2014 = DATEPART(YEAR, DateAccepted))
ORDER BY ClientName;
