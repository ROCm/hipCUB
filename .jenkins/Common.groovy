import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;
import groovy.transform.Field

@Field boolean formatCheck = false

@Field def getCompileCommand() =
{
    platform, project->

    project.paths.construct_build_prefix()
    
    def command 

    if(platform.jenkinsLabel.contains('hip-clang'))
    {
        command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                """
    }
    else
    {
        command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hcc ${project.paths.build_command}
                """
    }

    return command
}

// @Field def compileCommand =
// {
//     platform, project->

//     project.paths.construct_build_prefix()
    
//     def command 

//     if(platform.jenkinsLabel.contains('hip-clang'))
//     {
//         command = """#!/usr/bin/env bash
//                 set -x
//                 cd ${project.paths.project_build_prefix}
//                 LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
//                 """
//     }
//     else
//     {
//         command = """#!/usr/bin/env bash
//                 set -x
//                 cd ${project.paths.project_build_prefix}
//                 LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hcc ${project.paths.build_command}
//                 """
//     }

//     platform.runCommand(this, command)
// }

@Field def testCommand =
{
    platform, project->

    def command

    if(platform.jenkinsLabel.contains('centos'))
    {
        command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make -j4
                sudo ctest --output-on-failure
            """
    }
    else
    {
        command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make -j4
                ctest --output-on-failure
            """
    }

    platform.runCommand(this, command)
}

@Field def packageCommand =
{
    platform, project->

    def command
    
    if(platform.jenkinsLabel.contains('centos'))
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                rm -rf package && mkdir -p package
                mv *.rpm package/
                rpm -qlp package/*.rpm
              """
        
        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")
    }
    else if(platform.jenkinsLabel.contains('hip-clang'))
    {
        packageCommand = null
    }
    else
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                rm -rf package && mkdir -p package
                mv *.deb package/
                dpkg -c package/*.deb
              """        
        
        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
    }
}

return this