"""Windows no-follow handles; directory handles prevent ancestor renames."""

import sys

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    from pathlib import Path

    class FileInformation(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation", wintypes.FILETIME),
            ("access", wintypes.FILETIME),
            ("write", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]

    def _information(handle: int) -> FileInformation:
        info = FileInformation()
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandle
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(FileInformation))
        query.restype = wintypes.BOOL
        if not query(handle, ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return info

    def handle_identity(handle: int) -> tuple[int, int]:
        info = _information(handle)
        return int(info.volume), (int(info.index_high) << 32) | int(info.index_low)

    def open_handle(path: Path, *, directory: bool) -> int:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create.restype = wintypes.HANDLE
        handle = create(
            str(path),
            0x80 if directory else 0x80000000,  # FILE_READ_ATTRIBUTES or GENERIC_READ.
            3 if directory else 1,  # No FILE_SHARE_DELETE; files also deny new writers.
            None,
            3,  # OPEN_EXISTING
            0x00200000 | (0x02000000 if directory else 0),
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

        try:
            info = _information(handle)
            if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
                raise OSError("links and unexpected file types are forbidden")
            if not directory and info.links != 1:
                raise OSError("hard links are forbidden")
        except BaseException:
            close_handle(handle)
            raise
        return handle

    def close_handle(handle: int) -> None:
        close = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        close(handle)
