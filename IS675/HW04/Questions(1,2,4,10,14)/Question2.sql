--Change the format of the output from question #1 to add column aliases, format the phone number, 
 --and sort the result table by ClientZip.

 SELECT	ClientID 'ClientId',
		ClientName 'Client Name ',
		ClientCity 'Client Billing City ',
		ClientState 'Client Billing State',
		ClientZip 'Client Billing Zip',
	    STUFF(STUFF(STUFF(Phone,1,0,'('),6,0,') '),11,0,'-') as 'Client Billing Phone'
 FROM Client
 WHERE ClientCity = 'Reno'
 ORDER BY ClientZip;