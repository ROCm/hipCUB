#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

import java.nio.file.Path;

hipCUBCI:
{

    def hipcub = new rocProject('hipCUB')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx803 && centos && hip-clang', 'gfx900 && ubuntu && hip-clang', 'gfx906 && ubuntu && hip-clang'], hipcub)

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                    """
        
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def command

        if(platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
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

    def packageCommand = null

    buildProject(hipcub, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

