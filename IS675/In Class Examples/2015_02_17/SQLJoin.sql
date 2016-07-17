

CREATE TABLE Ord
(OrderID	char(10),
 OrderDate  datetime,
 CustID		char(10),
 DueDate	datetime);

 CREATE TABLE Cust
 (CustID	char(10),
  CustomerName	Varchar(35));

INSERT INTO Ord VALUES
(100, '02/06/2015', 1234, '02/11/2015'),
(200, '02/09/2015', 6773, '02/17/2015'),
(300, '02/18/2015', 1234, '03/2/2015');

INSERT INTO Cust VALUES
(1234, 'John Smith'),
(2555, 'Jane Doe'),
(6773, 'Bertie Wooster'),
(8372, 'Martin Cheng');

----------------------------------------------------------
SELECT
	*
FROM
	ord
;

SELECT
	*
FROM
	cust
;

SELECT
	*
FROM
	ord,
	cust
;

-- old, not so good
SELECT
	*
FROM
	ord,
	cust
WHERE
	ord.custid = cust.custid
;

-- the modern, better way
SELECT
	*
FROM
	ord
INNER JOIN
	cust on ord.custid = cust.custid
;

SELECT
	cust.CustomerName,
	ISNULL(ord.orderID, 'No Order') 'Order ID',
	ord.OrderDate
FROM
	ord
INNER JOIN
	cust
ON
	ord.custID = cust.custID
ORDER BY
	cust.customername
;

SELECT
	cust.CustomerName,
	ISNULL(ord.orderID, 'No Order') 'Order ID',
	ord.OrderDate
FROM
	ord
RIGHT OUTER JOIN
	cust
ON
	ord.custID = cust.custID
ORDER BY
	cust.customername
;

SELECT
	cust.CustomerName,
	ISNULL(ord.orderID, 'No Order') 'Order ID',
	ord.OrderDate
FROM
	cust
LEFT OUTER JOIN
	ord
ON
	ord.custID = cust.custID
ORDER BY
	cust.customername
;

