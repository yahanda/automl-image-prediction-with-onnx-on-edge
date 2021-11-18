The main solution file is OnnxRuntime.CSharp.sln. This includes desktop and Xamarin mobile projects.
OnnxRuntime.DesktopOnly.CSharp.sln is a copy of that with all the mobile projects removed. 

If changes are required, either update the main solution first and copy the relevant changes across,
 or copy the entire file and remove the mobile projects (anything with iOS, Android or Droid in the name). 