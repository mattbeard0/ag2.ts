# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import sys
import tempfile
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ... import Tool

__all__ = ["PythonLocalExecutionTool"]

with optional_import_block():
    import aiofiles


@require_optional_import(
    [
        "aiofiles",
    ],
    "code-execution",
)
@export_module("autogen.tools.experimental")
class PythonLocalExecutionTool(Tool):
    """Executes Python code locally and returns the result. Defaults to using a virtual environment (venv) for isolation.

    **CAUTION**: This tool will execute code in your local environment, which can be dangerous if the code is untrusted.
    """

    def __init__(self, *, use_venv: bool = True, venv_path: Optional[str] = None, timeout: int = 30) -> None:
        """
        Initialize the PythonLocalExecutionTool.

        **CAUTION**: This tool will execute code in your local environment, which can be dangerous if the code is untrusted.

        Args:
            use_venv: Whether to use a Python virtual environment for execution. Defaults to True.
            venv_path: Custom path for virtual environment. If None, creates a temp venv called 'ag2_python_exec_venv_'.
            timeout: Maximum execution time allowed in seconds, will raise a TimeoutError exception if exceeded.
        """
        # Store configuration parameters for use in tool function
        self.use_venv = use_venv
        self.venv_path = venv_path
        self.timeout = timeout

        # Pydantic model to contain the code and list of libraries to execute
        class CodeExecutionRequest(BaseModel):
            code: Annotated[str, Field(description="Python code to execute")]
            libraries: Annotated[list[str], Field(description="List of libraries to install before execution")]

        # The tool function, this is what goes to the LLM
        async def execute_python_locally(
            code_execution_request: Annotated[CodeExecutionRequest, "Python code and the libraries required"],
        ) -> Any:
            """
            Executes Python code locally and returns the result.

            Args:
                code_execution_request (CodeExecutionRequest): The Python code and libraries to execute
                use_venv_param (bool): Whether to use a virtual environment
                venv_path_param (str): Custom path for virtual environment
                timeout_param (int): Maximum execution time in seconds
            """
            libraries = code_execution_request.libraries or []

            # Choose execution method based on configuration
            if self.use_venv:
                return await execute_in_venv(
                    code=code_execution_request.code,
                    libraries=libraries,
                    venv_path=self.venv_path,
                    timeout=self.timeout,
                )
            else:
                return await execute_directly(
                    code=code_execution_request.code, libraries=libraries, timeout=self.timeout
                )

        async def execute_in_venv(
            code: str, libraries: list[str], venv_path: Optional[str], timeout: int
        ) -> dict[str, Any]:
            """Execute Python code in a virtual environment."""
            import shutil

            # Create a unique temporary venv if path not provided
            temp_venv_dir = None
            venv_python = None

            try:
                # Create a temporary directory for the script
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Write the code to a temporary file
                    script_path = os.path.join(temp_dir, "script.py")
                    async with aiofiles.open(script_path, "w") as f:
                        await f.write(code)

                    # Determine venv path and whether we need to create it
                    if venv_path is None:
                        temp_venv_dir = tempfile.mkdtemp(prefix="ag2_python_exec_venv_")
                        venv_path = temp_venv_dir

                        # Create the virtual environment
                        venv_proc = await asyncio.create_subprocess_exec(
                            sys.executable,
                            "-m",
                            "venv",
                            venv_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        try:
                            stdout, stderr = await asyncio.wait_for(venv_proc.communicate(), timeout=timeout)
                        except asyncio.TimeoutError:
                            venv_proc.kill()
                            return {
                                "success": False,
                                "error": f"Virtual environment creation timed out after {timeout} seconds",
                            }

                        if venv_proc.returncode != 0:
                            return {
                                "success": False,
                                "error": f"Failed to create virtual environment: {stderr.decode('utf-8', errors='replace')}",
                            }

                    # Determine the Python executable path in the venv
                    if os.name == "nt":  # Windows
                        venv_python = os.path.join(venv_path, "Scripts", "python.exe")
                    else:  # Linux/Mac
                        venv_python = os.path.join(venv_path, "bin", "python")

                    # Verify venv Python executable exists
                    if not os.path.exists(venv_python):
                        return {
                            "success": False,
                            "error": f"Virtual environment Python executable not found at {venv_python}",
                        }

                    # Install dependencies if any
                    if libraries:
                        pip_proc = await asyncio.create_subprocess_exec(
                            venv_python,
                            "-m",
                            "pip",
                            "install",
                            *libraries,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        try:
                            stdout, stderr = await asyncio.wait_for(pip_proc.communicate(), timeout=timeout)
                        except asyncio.TimeoutError:
                            pip_proc.kill()
                            return {
                                "success": False,
                                "error": f"Installation of package dependency timed out after {timeout} seconds",
                            }

                        if pip_proc.returncode != 0:
                            return {
                                "success": False,
                                "error": f"Failed to install package dependencies: {stderr.decode('utf-8', errors='replace')}",
                            }

                    # Execute the script using the venv Python
                    exec_proc = await asyncio.create_subprocess_exec(
                        venv_python, script_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )

                    try:
                        stdout, stderr = await asyncio.wait_for(exec_proc.communicate(), timeout=timeout)
                    except asyncio.TimeoutError:
                        exec_proc.kill()
                        return {"success": False, "error": f"Execution timed out after {timeout} seconds"}

                    return {
                        "success": exec_proc.returncode == 0,
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                        "returncode": exec_proc.returncode,
                    }

            except Exception as e:
                return {"success": False, "error": f"Execution error: {str(e)}"}
            finally:
                # Clean up temporary venv if we created one
                if temp_venv_dir and os.path.exists(temp_venv_dir):
                    with contextlib.suppress(Exception):
                        shutil.rmtree(temp_venv_dir)

        async def execute_directly(code: str, libraries: list[str], timeout: int) -> dict[str, Any]:
            """Execute Python code directly in the current Python environment.

            **CAUTION**: This is not recommended for untrusted code.
            """
            try:
                # Create a temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Write the code to a temporary file
                    script_path = os.path.join(temp_dir, "script.py")
                    async with aiofiles.open(script_path, "w") as f:
                        await f.write(code)

                    # Install dependencies if any
                    if libraries:
                        try:
                            pip_proc = await asyncio.create_subprocess_exec(
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                *libraries,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )

                            try:
                                stdout, stderr = await asyncio.wait_for(pip_proc.communicate(), timeout=timeout)
                            except asyncio.TimeoutError:
                                pip_proc.kill()
                                return {
                                    "success": False,
                                    "error": f"Installation of package dependency timed out after {timeout} seconds",
                                }

                            if pip_proc.returncode != 0:
                                return {
                                    "success": False,
                                    "error": f"Failed to install package dependencies: {stderr.decode('utf-8', errors='replace')}",
                                }

                        except Exception as e:
                            return {"success": False, "error": f"Failed to install package dependencies: {str(e)}"}

                    # Execute the script
                    exec_proc = await asyncio.create_subprocess_exec(
                        sys.executable, script_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                    )

                    try:
                        stdout, stderr = await asyncio.wait_for(exec_proc.communicate(), timeout=timeout)
                    except asyncio.TimeoutError:
                        exec_proc.kill()
                        return {"success": False, "error": f"Execution timed out after {timeout} seconds"}

                    return {
                        "success": exec_proc.returncode == 0,
                        "stdout": stdout.decode("utf-8", errors="replace"),
                        "stderr": stderr.decode("utf-8", errors="replace"),
                        "returncode": exec_proc.returncode,
                    }

            except Exception as e:
                return {"success": False, "error": f"Execution error: {str(e)}"}

        super().__init__(
            name="python_execute_local",
            description="Executes Python code locally and returns the result.",
            func_or_tool=execute_python_locally,
        )
