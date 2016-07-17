using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using com.FirstWorldDoorbell.KeypadSerial;
using com.FirstWorldDoorbell.Alert;

namespace ConsoleApplication1
{
	class MainClass
	{
		public static void Main(string[] args)
		{

			KeypadSerialReader reader = new KeypadSerialReader();
			reader.DoStuff();

			while (true) {
				while (!reader.EntryCompleted())
				{ 
				}

				String entry = reader.GetEntry();
				int code = Int16.Parse(entry);

				Console.WriteLine("Sending code: " + code);
				Alert.SendAlert(Alert.SAMPLE_CUSTOMER_EMAIL, code);

				reader.Reset();
			}

			reader.Finalize();
		}

	}
}
