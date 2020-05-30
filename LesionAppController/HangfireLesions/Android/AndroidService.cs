using System;
using System.Collections.Generic;
using System.Text;

namespace HangfireLesions.AndroidService
{

    public interface IAndroidService 
    {
        void RunAndroidTask();
    }
    public class AndroidService : IAndroidService
    {
        public void RunAndroidTask()
        {
            Console.WriteLine("This is a test.");
        }
    }
}
