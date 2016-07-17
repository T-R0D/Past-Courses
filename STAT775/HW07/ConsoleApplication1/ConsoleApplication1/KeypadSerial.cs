using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO.Ports;

namespace com.FirstWorldDoorbell.KeypadSerial
{
	class KeypadSerialReader
	{
		SerialPort arduinoPort;

		String entry;
		bool entryCompleted = false;



		public void DoStuff()
		{

			Console.Out.WriteLine("Let's try this out!");

			arduinoPort = new SerialPort("COM15");

			arduinoPort.BaudRate = 9600;
			arduinoPort.Parity = Parity.None;
			arduinoPort.StopBits = StopBits.One;
			arduinoPort.DataBits = 8;
			arduinoPort.Handshake = Handshake.None;

			arduinoPort.DataReceived += new SerialDataReceivedEventHandler(DataReceivedHandler);

			arduinoPort.Open();


			//Console.WriteLine("Press any key to continue...");
			//Console.WriteLine();
			//Console.ReadKey();
			//arduinoPort.Close();
		}

		public void Finalize()
		{
			arduinoPort.Close();
		}

		public bool EntryCompleted()
		{
			return entryCompleted;
		}

		public void Reset()
		{
			entryCompleted = false;
			entry = "";
		}

		public String GetEntry()
		{
			return entry;
		}


		private void DataReceivedHandler(object sender, SerialDataReceivedEventArgs e)
		{
			SerialPort sp = (SerialPort) sender;
			String indata = sp.ReadExisting();

			if (entryCompleted != true) { 
				indata = indata.Trim();

				if (indata.Equals("*"))
				{
					entryCompleted = true;
					Console.WriteLine("Got an entry");
				}
				else
				{
					entry += indata;
				}
			}

			//Console.WriteLine(entry);

			//Debug.Print("Data Received:");
			//Debug.Print(indata);
		}

	}
}
