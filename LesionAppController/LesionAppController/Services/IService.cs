using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace LesionAppController.Services
{
    interface IService
    {
        void RunModel();

        void MoveModel();

        void BuildMobileApp();

        void UploadToGoogleDrive();

        void PushUpdateApp();

        void MyTask();
    }
}
