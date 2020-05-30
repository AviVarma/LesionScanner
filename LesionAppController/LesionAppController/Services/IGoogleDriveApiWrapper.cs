using Google.Apis.Drive.v3;
using Google.Apis.Drive.v3.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace LesionAppController.Services
{
    interface IGoogleDriveApiWrapper
    {
        DriveService InitializeDrive();

        string[] ListFiles(DriveService _service);

        File UploadFile(DriveService _service, string _uploadFile, string _parent, string _descrp = "Updated APK for Lesion Scanner");

        void DeleteFile(DriveService _service, string id);
    }
}
