DROP TABLE
	xtblVendor
;

CREATE TABLE
	xtblVendor(
		  VendorID		char(4)
		, VendorName	varchar(30)	NOT NULL
		, FirstBuyDate	datetime
		CONSTRAINT pkVendor
			PRIMARY KEY (vendorID)
	)
;

INSERT INTO
	xtblVendor
VALUES
	('7819', 'Martinson Concrete and Supply', '04/15/2013')
;
INSERT INTO
	xtblVendor
VALUES
	('2745', 'Johnson Plating', '14-oct-2014')
;
INSERT INTO
	xtblVendor
VALUES
	('0062', 'Evergreen Surface Products', '07-12-2012')
;
INSERT INTO
	xtblVendor
VALUES
	(0062, 'Touchstone Materials', '05-16-13');SELECT
	VendorID
	, VendorName
FROM
	xtblVendor
WHERE	VendorID = '0062';