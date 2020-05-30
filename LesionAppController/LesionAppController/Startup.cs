using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Hangfire;
using Hangfire.MemoryStorage;
using System;
using LesionAppController.Services;
//using Microsoft.AspNetCore.Builder;
//using Microsoft.Extensions.DependencyInjection;

namespace LesionAppController
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }


        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddHangfire(config =>
            {
                config.UseMemoryStorage();
            });

            services.AddScoped<IService, Service>();

        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            app.UseHangfireDashboard();
            app.UseHangfireServer();

            var serviceClass = new Service();
            IService MyService = serviceClass;

          

            RecurringJob.AddOrUpdate("Train Model", ()=> MyService.RunModel(), Cron.Daily);

            RecurringJob.AddOrUpdate("Move Model", () => MyService.MoveModel(), Cron.Daily);

            RecurringJob.AddOrUpdate("Make Mobile App", () => MyService.BuildMobileApp(), Cron.Daily);

            RecurringJob.AddOrUpdate("upload APK", () => MyService.UploadToGoogleDrive(), Cron.Daily);

            RecurringJob.AddOrUpdate("Email Updates", () => MyService.PushUpdateApp(), Cron.Daily);

            // Test job for debugging. Use to check if hangfire setup is correct.
            //RecurringJob.AddOrUpdate("Example Execution", () => MyService.MyTask(),Cron.Minutely);
        }
    }
}
