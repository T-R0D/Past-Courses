GO
CREATE TRIGGER unit_insert_log
   ON a_unit FOR INSERT AS   
BEGIN
   INSERT INTO a_UnitLog(
     userName,
     logDate,
	 SerialNumber)
   VALUES(
     USER,
     GETDATE(),
	 (SELECT serialNumber from INSERTED) )	  
END

INSERT INTO a_Unit VALUES(35649,212,getDate());
select * from a_UnitLog


