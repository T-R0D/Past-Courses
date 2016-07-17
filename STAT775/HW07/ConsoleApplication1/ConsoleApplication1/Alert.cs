using System;
using System.Net;
using System.Net.Mail;
using System.Net.Mime;
using System.Threading;
using System.ComponentModel;

using System.Collections.Generic;

namespace com.FirstWorldDoorbell.Alert
{
    public class Alert
    {
        static readonly String SMTP_HOST = "smtp.gmail.com";
        static readonly int SMTP_PORT = 587;
        static readonly String TERENCE_EMAIL = "thenriod@gmail.com";
        static readonly String TERENCE_PASS = "Hurd13110";
        static readonly String DOORBELL_CO_EMAIL = TERENCE_EMAIL;
        public static readonly String SAMPLE_CUSTOMER_EMAIL = "7753047063@vtext.com";
        static readonly Dictionary<int, String> SAMPLE_ASSOCIATIONS =
            new Dictionary<int,string> {
                {666, "Hide from Satan!"},
                {777, "Collect your winnings!"},
                {420, "Blaze it!"},
                {867, "5-3-oh-ni-ee-ine!"},
                {888, "Hells yeah! It's T-R0D! Let him in!"},
                {555, "Bandith! I hate that guy! Hide!"},
                {123, "Answer the door! The judge just declared you THE winner!"}
            };

        //public static void Main(string[] args)
        //{
        //	SendAlert(SAMPLE_CUSTOMER_EMAIL, 888);
        //}

        public static void SendAlert(String customerTextEmail, int code)
        {
            SmtpClient smtpClient = new SmtpClient(SMTP_HOST, SMTP_PORT);
            smtpClient.EnableSsl = true;
            smtpClient.UseDefaultCredentials = false;
            smtpClient.Credentials = new System.Net.NetworkCredential(TERENCE_EMAIL, TERENCE_PASS);

            MailAddress doorbellCoEmail = new MailAddress(DOORBELL_CO_EMAIL);
            MailAddress customerEmail = new MailAddress(customerTextEmail);

            MailMessage message = new MailMessage(from: doorbellCoEmail, to: customerEmail);
            message.Subject = null;
            message.SubjectEncoding = System.Text.Encoding.UTF8;
        
            // TODO: add our super-cool functionality to generate an informative message body
            message.Body = BuildAlertMessage(code, SAMPLE_ASSOCIATIONS);
            message.BodyEncoding = System.Text.Encoding.UTF8;

            smtpClient.Send(message);
            message.Dispose();
        }

        public static String BuildAlertMessage(int code, Dictionary<int, String> associations)
        {
            String alertMessage = "SOMEONE IS AT THE DOOR!\n";
            alertMessage += "Code: " + code + "\n";
            alertMessage += "Recommendation: ";
            String recommendation = "";
            if (!associations.TryGetValue(code, out recommendation))
            {
                recommendation = "None.";
            }
            alertMessage += recommendation;

            return alertMessage;
        }
    }
}


