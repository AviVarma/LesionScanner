using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using MailKit.Net.Smtp;
using MimeKit;
using Google.Apis.Auth.OAuth2;
using Google.Apis.Drive.v3;
using Google.Apis.Services;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace LesionAppController.Services
{
    public class Service : IService
    {
        private string trainDirectory =
            @"..\..\base_dir\train_dir";

        private string modelDumpFolderPath =
            @"..\..\model_dump\model.tflite";

        private string assetsFolderPath =
            @"..\..\lesion_scanner_app\app\src\main\assets\model.tflite";

        private string androidApkOutputPath =
            @"..\..\lesion_scanner_app\app\build\outputs\apk\debug\app-debug.apk";

        private string apkDumpFolderPath =
            @"..\..\apk_dump\app-debug.apk";

        //private int numImagesLastCheched = 38569;

        /**
         * Check if there has been any additions to the training directory.
         * If there has been, then re-tarin the ML model. Or else, log the end condtition executed.
         */
        public void RunModel()
        {
            string path = @"./Services/env_variables.txt";
            string value = File.ReadAllText(path);
            int numImagesLastCheched = int.Parse(value);
            int currentImagesNum = 
                Directory.EnumerateFiles(trainDirectory, "*.*", SearchOption.AllDirectories).Count();
            if (numImagesLastCheched == currentImagesNum) 
            {
                Console.WriteLine("No new images, no need to re-train.");
            }
            else
            {
                string new_value = value.Replace(value, currentImagesNum.ToString());
                File.WriteAllText(path, new_value);

                Run_cmd(@"..\..\ModelBuilder\ModelTrainer.py",
                    @"C:\Users\aviva\AppData\Local\Programs\Python\Python37\python.exe"); // Please replace this whith yourlocation of the python executable.
            }
        }

        /**
         * Check if there is a model in model_dump then accroding to the result move the model
         * into the mobile app project.
         */
        public void MoveModel() 
        {
            // check the target path if it already contains the model.
            // delete the old model for the new model.
            if (IsDirectoryEmpty(@"..\..\lesion_scanner_app\app\src\main\assets\") == false)
            {
                if (File.Exists(assetsFolderPath))
                {
                    File.Delete(assetsFolderPath);
                }
            }

            if (File.Exists(modelDumpFolderPath) == false)
            {
                Console.WriteLine("No model found in model dump folder. Aborting Move Model Function.");
            }
            else 
            {
                Console.WriteLine("Moving model to build mobile app.");
                MoveFile(modelDumpFolderPath, assetsFolderPath);
            }
        }

        public void BuildMobileApp()
        {
            if (File.Exists(assetsFolderPath) == true)
            {
                string rootDir = Directory.GetCurrentDirectory();
                Console.WriteLine(rootDir);
                string targetDir = @"..\..\lesion_scanner_app";
                string executionDir = @"D:\Documents\Year 3\Third Year Project\Project\lesion_scanner_app\gradlew.bat"; // This needs to be an absolute path!
                Directory.SetCurrentDirectory(targetDir);
                Run_cmd("assembleDebug", executionDir);
                Directory.SetCurrentDirectory(rootDir);

                File.Copy(androidApkOutputPath,
                    apkDumpFolderPath);
            }
            else 
            {
                Console.WriteLine("No model found in assets folder. Aborting Mobile App Builder.");
            }
        }

        /**
        1. check google drive folder is empty.
        2. if empty: upload the apk file to the folder.
        3. If not empty: delete all the contents of the folder.
        */
        public void UploadToGoogleDrive()
        {

            if (File.Exists(apkDumpFolderPath))
            {
                var googleWrapperClass = new GoogleDriveApiWrapper();
                IGoogleDriveApiWrapper googleWrapper = googleWrapperClass;

                DriveService service = googleWrapper.InitializeDrive();
                string[] fileIDs = googleWrapper.ListFiles(service);

                if (fileIDs.Length > 0)
                {
                    for (int i = 0; i < fileIDs.Length; i++)
                    {
                        googleWrapper.DeleteFile(service, fileIDs[i]);
                    }
                }
                var responseToServer = googleWrapper.UploadFile(service, apkDumpFolderPath, "17qK-EKoR3KCZv-xDvJ8kt0bEXF0m8Jx7");
                Console.WriteLine("Process completed--- Response--" + responseToServer);
            }
            else 
            {
                Console.WriteLine("Apk not found! Aborting upload to Google Drive.");
            }
        }

        public void PushUpdateApp() 
        {

            if (IsDirectoryEmpty(@"..\..\apk_dump") == false)
            {
                /**
                 * Read from the mail_list.json and notify everyone of the new updated apk.
                 */
                var json = File.ReadAllText(@".\Services\mail_list.json");
                string[] array = JsonConvert.DeserializeObject<string[]>(json);

                foreach (string address in array)
                {
                    Push_email(address);
                }

                Console.WriteLine("E-mail sent to all authorized users.");

                /**
                 * This is the end of the pipeline.
                 * Hence all the "Dump" folders will be cleaned for future runs.
                 */

                // clean up the assets folder:
                if (File.Exists(assetsFolderPath))
                {
                    File.Delete(assetsFolderPath);
                }

                // Delete the apk from the dump folder:
                if (File.Exists(apkDumpFolderPath))
                {
                    File.Delete(apkDumpFolderPath);
                }
            }
            else 
            {
                Console.WriteLine("No APK found, app is up to date.");
            }
        }


        /**
         * Test function to check if hangfre has been setup correctly.
         */
        public void MyTask()
        {
            Console.WriteLine("This is a quick test! --- Function executed!");
        }

        private bool IsDirectoryEmpty(string path)
        {
            return !Directory.EnumerateFileSystemEntries(path).Any();
        }

        private void MoveFile(string sourcePath, string targetPath)
        {
            try
            {
                // Move the file.
                File.Move(sourcePath, targetPath);
                Console.WriteLine("{0} was moved to {1}.", sourcePath, targetPath);

                // See if the original exists in the source path.
                if (File.Exists(sourcePath))
                {
                    Console.WriteLine("The original file still exists, which is unexpected.");
                }
                else
                {
                    Console.WriteLine("The original file no longer exists, which is expected.");
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("The process failed: {0}", e.ToString());
            }
        }


        /**
         * take in a valid email address and send the hard-coded mail to the user.
         * 
         * @param address A valid e-mail address
         */
        private void Push_email(string address) 
        {
            MimeMessage message = new MimeMessage();

            MailboxAddress from = new MailboxAddress("Admin",
            "avi.varma@ymail.com");
            message.From.Add(from);

            MailboxAddress to = new MailboxAddress("User",
            address); // this should be grabbed from a json file.
            message.To.Add(to);

            message.Subject = "New Lesion Scanner Update available";
            BodyBuilder bodyBuilder = new BodyBuilder();
            bodyBuilder.TextBody = "Hello user, I have an updated version of the skin lesion android app." +
                " It's APK can be found with the link below. This update has a an improoved version of the prediction software. " +
                "Please install the updated app for inproove prediciton accuracy.\n" +
                "https://drive.google.com/drive/folders/17qK-EKoR3KCZv-xDvJ8kt0bEXF0m8Jx7?usp=sharing";
            message.Body = bodyBuilder.ToMessageBody();

            SmtpClient client = new SmtpClient();
            client.Connect("smtp.mail.yahoo.com", 465, true);
            client.Authenticate("avi.varma@ymail.com", "slmffzlcavlrwgpu");

            client.Send(message);
            client.Disconnect(true);
            client.Dispose();
        }

        /**
         * Run the executable and the command in the system console.
         * 
         * @param cmd Provide the command here.
         * @param executablePath Provide the executable here. 
         */
        private static void Run_cmd(string cmd, string executablePath)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = executablePath;
            start.Arguments = string.Format("{0}", cmd);
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    Console.Write(result);
                }
            }
        }
    }
}
