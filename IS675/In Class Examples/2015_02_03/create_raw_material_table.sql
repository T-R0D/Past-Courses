DROP TABLE
	xtblRawMaterial
;

CREATE TABLE
	xtblRawMaterial(
		  MaterialID		char(3)
		, Description		varchar(50)
		, UnitOfMeasure		char(8)
		, StandardPrice		money		CHECK (StandardPrice > 0)
		CONSTRAINT pkRawMaterial
			PRIMARY KEY (MaterialID)
	)
;

INSERT INTO
	xtblRawMaterial
VALUES
	  ('255', 'Concrete Overlay Polymer', 'pound', 13.75)
	, ('240', 'Mortar Mix', null, .23)
	, ('271', 'Graphite Ismolded Sheet', 'each', 46.70)
;
