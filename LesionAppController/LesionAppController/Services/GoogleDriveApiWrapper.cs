using Google.Apis.Auth.OAuth2;
using Google.Apis.Drive.v3;
using Google.Apis.Services;
using Google.Apis.Util.Store;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using File = Google.Apis.Drive.v3.Data.File;

namespace LesionAppController.Services
{
    public class GoogleDriveApiWrapper : IGoogleDriveApiWrapper
    {
        /**
         * Construct an instance of the google drive service so that the account provided can be accessed.
         */
        public DriveService InitializeDrive()
        {
            string[] scopes = new string[] { DriveService.Scope.Drive,
                               DriveService.Scope.DriveFile,};
            var clientId = "249038161247-qlglhi108bco6ifb3k88r5v6etusfo0k.apps.googleusercontent.com";// From https://console.developers.google.com  
            var clientSecret = "HPBkwy3hcoMEzt_R_j0v-DxO";// From https://console.developers.google.com  
            // here is where we Request the user to give us access, or use the Refresh Token that was previously stored in %AppData%  
            var credential = GoogleWebAuthorizationBroker.AuthorizeAsync(new ClientSecrets
            {
                ClientId = clientId,
                ClientSecret = clientSecret
            }, scopes,
            Environment.UserName, CancellationToken.None, new FileDataStore("MyAppsToken")).Result;
            //Once consent is recieved, your token will be stored locally on the AppData directory, so that next time you wont be prompted for consent.   

            DriveService service = new DriveService(new BaseClientService.Initializer()
            {
                HttpClientInitializer = credential,
                ApplicationName = "MyAppName",

            });
            service.HttpClient.Timeout = TimeSpan.FromMinutes(100);
            return service;
        }

        /**
         * 
         * Create a string array that contains all the contents of the googl drive and reurn it.
         * @param _service The initialized google drive service that will be querired.
         * @return string[] list of items from the google drive.
         */
        public string[] ListFiles(DriveService _service)
        {
            FilesResource.ListRequest listRequest = _service.Files.List();
            listRequest.PageSize = 10;
            listRequest.Fields = "nextPageToken, files(id, name)";


            IList<File> files = listRequest.Execute().Files;
            string[] fileIDs = new string[files.Count - 1];
            int i = 0;
            Console.WriteLine("Files:");
            if (files != null && files.Count > 0)
            {
                foreach (var file in files)
                {
                    if (file.Id != "17qK-EKoR3KCZv-xDvJ8kt0bEXF0m8Jx7")
                    {
                        Console.WriteLine("{0} ({1})", file.Name, file.Id);
                        fileIDs[i] = file.Id;
                        i++;
                    }
                }
                return fileIDs;
            }
            else
            {
                Console.WriteLine("No files found.");
            }
            Console.Read();
            return null;
        }

        /**
         * The function will take in the file and the parent directory in which you want the file to be in within the
         * google drive and upload it. The mime and other mota data is collected using the private function
         * GetMimeType(string fileName).
         * 
         * @param _service The initialized google drive service that will be querired.
         * @param _uploadFile The file you wish to upload to the google drive.
         * @param _parent The parent dierctory that the file should be placed in within the google drive.
         * @param _descrp each file being uploaded needs a description. This is filled in by default.
         * @return Return the response message from the server.
         */
        public File UploadFile(DriveService _service, string _uploadFile, string _parent, string _descrp = "Updated APK for Lesion Scanner")
        {

            if (System.IO.File.Exists(_uploadFile))
            {
                File body = new File();
                body.Name = Path.GetFileName(_uploadFile);
                body.Description = _descrp;
                body.MimeType = GetMimeType(_uploadFile);
                body.Parents = new List<string> { _parent };
                byte[] byteArray = System.IO.File.ReadAllBytes(_uploadFile);
                MemoryStream stream = new System.IO.MemoryStream(byteArray);
                try
                {
                    FilesResource.CreateMediaUpload request = _service.Files.Create(body, stream, GetMimeType(_uploadFile));
                    request.SupportsTeamDrives = true;
                    // You can bind event handler with progress changed event and response recieved(completed event)
                    request.Upload();
                    return request.ResponseBody;
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message, "Error Occured");
                    return null;
                }
            }
            else
            {
                Console.WriteLine("The file does not exist.", "404");
                return null;
            }
        }


        /**
         * The file provided will be searched for and deleted from the google drive.
         * 
         * @param _service The initialized google drive service that will be querired.
         * @param id The file id that needs to be deleted.
         */
        public void DeleteFile(DriveService _service, string id)
        {
            _service.Files.Delete(id).Execute();
            Console.WriteLine("file " + id + " deleted!");
        }

        /**
         * Private function which will find the metatadata of the provided file and return it.
         * 
         * @param fileName Path to the file which needs to be searched.
         * @return mimeType return the file's mime.
         */
        private string GetMimeType(string fileName)
        {
            string mimeType = "application/unknown";
            string ext = Path.GetExtension(fileName).ToLower();
            Microsoft.Win32.RegistryKey regKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(ext);
            if (regKey != null && regKey.GetValue("Content Type") != null)
                mimeType = regKey.GetValue("Content Type").ToString();
            return mimeType;
        }
    }
}
