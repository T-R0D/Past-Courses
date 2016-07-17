

CREATE TABLE Ord1
(OrderID	char(10),
 OrderDate  datetime,
 CustID		char(10),
 DueDate	datetime);

 INSERT INTO Ord1 VALUES
('100', '02/06/2015', '1234', '02/11/2015'),
('200', '02/09/2015', '6773', '02/17/2015'),
('300', '02/18/2015', '1234', '03/02/2015'),
('400', '01/27/2015', '2555', '02/02/2015'),
('500', '02/12/2015', '8989', '02/22/2015'),
('600', '01/28/2015', '2555', '01/31/2015'),
('700', '02/05/2015', '2555', '02/13/2015');


SELECT
	*
FROM
	ord1
INNER JOIN
	cust ON
		ord1.custid = cust.custid
;

SELECT
	orderid,
	orderdate,
	ord1.custid,
	customername
FROM
	ord1
LEFT OUTER JOIN
	cust ON
		ord1.custid = cust.custid
;


SELECT
	orderid,
	orderdate,
	ord1.custid,
	customername
FROM
	ord1
FULL OUTER JOIN
	cust ON
		ord1.custid = cust.custid
;


SELECT
	orderid,
	orderdate,
	ord1.custid
FROM
	ord1
FULL OUTER JOIN
	cust ON
		ord1.custid = cust.custid
WHERE
	cust.CustomerName = 'Jane Doe'
;

SELECT
	ord1.orderid,
	ord1.orderdate,
	cust.custid,
	ord1.DueDate,
	cust.custid,
	cust.CustomerName
FROM
	ord1
LEFT OUTER JOIN
	cust ON
		cust.custid IS NULL
;
