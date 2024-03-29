// Copyright 2019 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common/Assert.h"
#include "dawn_wire/client/Client.h"
#include "dawn_wire/client/Device.h"

#include <limits>

namespace dawn_wire { namespace client {

    bool Client::DoDeviceUncapturedErrorCallback(DawnErrorType errorType, const char* message) {
        switch (errorType) {
            case DAWN_ERROR_TYPE_NO_ERROR:
            case DAWN_ERROR_TYPE_VALIDATION:
            case DAWN_ERROR_TYPE_OUT_OF_MEMORY:
            case DAWN_ERROR_TYPE_UNKNOWN:
            case DAWN_ERROR_TYPE_DEVICE_LOST:
                break;
            default:
                return false;
        }
        mDevice->HandleError(errorType, message);
        return true;
    }

    bool Client::DoDevicePopErrorScopeCallback(uint64_t requestSerial,
                                               DawnErrorType errorType,
                                               const char* message) {
        return mDevice->PopErrorScope(requestSerial, errorType, message);
    }

    bool Client::DoBufferMapReadAsyncCallback(Buffer* buffer,
                                              uint32_t requestSerial,
                                              uint32_t status,
                                              uint64_t initialDataInfoLength,
                                              const uint8_t* initialDataInfo) {
        // The buffer might have been deleted or recreated so this isn't an error.
        if (buffer == nullptr) {
            return true;
        }

        // The requests can have been deleted via an Unmap so this isn't an error.
        auto requestIt = buffer->requests.find(requestSerial);
        if (requestIt == buffer->requests.end()) {
            return true;
        }

        auto request = std::move(requestIt->second);
        // Delete the request before calling the callback otherwise the callback could be fired a
        // second time. If, for example, buffer.Unmap() is called inside the callback.
        buffer->requests.erase(requestIt);

        const void* mappedData = nullptr;
        size_t mappedDataLength = 0;

        auto GetMappedData = [&]() -> bool {
            // It is an error for the server to call the read callback when we asked for a map write
            if (request.writeHandle) {
                return false;
            }

            if (status == DAWN_BUFFER_MAP_ASYNC_STATUS_SUCCESS) {
                if (buffer->readHandle || buffer->writeHandle) {
                    // Buffer is already mapped.
                    return false;
                }
                if (initialDataInfoLength > std::numeric_limits<size_t>::max()) {
                    // This is the size of data deserialized from the command stream, which must be
                    // CPU-addressable.
                    return false;
                }
                ASSERT(request.readHandle != nullptr);

                // The server serializes metadata to initialize the contents of the ReadHandle.
                // Deserialize the message and return a pointer and size of the mapped data for
                // reading.
                if (!request.readHandle->DeserializeInitialData(
                        initialDataInfo, static_cast<size_t>(initialDataInfoLength), &mappedData,
                        &mappedDataLength)) {
                    // Deserialization shouldn't fail. This is a fatal error.
                    return false;
                }
                ASSERT(mappedData != nullptr);

                // The MapRead request was successful. The buffer now owns the ReadHandle until
                // Unmap().
                buffer->readHandle = std::move(request.readHandle);
            }

            return true;
        };

        if (!GetMappedData()) {
            // Dawn promises that all callbacks are called in finite time. Even if a fatal error
            // occurs, the callback is called.
            request.readCallback(DAWN_BUFFER_MAP_ASYNC_STATUS_DEVICE_LOST, nullptr, 0,
                                 request.userdata);
            return false;
        } else {
            request.readCallback(static_cast<DawnBufferMapAsyncStatus>(status), mappedData,
                                 static_cast<uint64_t>(mappedDataLength), request.userdata);
            return true;
        }
    }

    bool Client::DoBufferMapWriteAsyncCallback(Buffer* buffer,
                                               uint32_t requestSerial,
                                               uint32_t status) {
        // The buffer might have been deleted or recreated so this isn't an error.
        if (buffer == nullptr) {
            return true;
        }

        // The requests can have been deleted via an Unmap so this isn't an error.
        auto requestIt = buffer->requests.find(requestSerial);
        if (requestIt == buffer->requests.end()) {
            return true;
        }

        auto request = std::move(requestIt->second);
        // Delete the request before calling the callback otherwise the callback could be fired a
        // second time. If, for example, buffer.Unmap() is called inside the callback.
        buffer->requests.erase(requestIt);

        void* mappedData = nullptr;
        size_t mappedDataLength = 0;

        auto GetMappedData = [&]() -> bool {
            // It is an error for the server to call the write callback when we asked for a map read
            if (request.readHandle) {
                return false;
            }

            if (status == DAWN_BUFFER_MAP_ASYNC_STATUS_SUCCESS) {
                if (buffer->readHandle || buffer->writeHandle) {
                    // Buffer is already mapped.
                    return false;
                }
                ASSERT(request.writeHandle != nullptr);

                // Open the WriteHandle. This returns a pointer and size of mapped memory.
                // On failure, |mappedData| may be null.
                std::tie(mappedData, mappedDataLength) = request.writeHandle->Open();

                if (mappedData == nullptr) {
                    return false;
                }

                // The MapWrite request was successful. The buffer now owns the WriteHandle until
                // Unmap().
                buffer->writeHandle = std::move(request.writeHandle);
            }

            return true;
        };

        if (!GetMappedData()) {
            // Dawn promises that all callbacks are called in finite time. Even if a fatal error
            // occurs, the callback is called.
            request.writeCallback(DAWN_BUFFER_MAP_ASYNC_STATUS_DEVICE_LOST, nullptr, 0,
                                  request.userdata);
            return false;
        } else {
            request.writeCallback(static_cast<DawnBufferMapAsyncStatus>(status), mappedData,
                                  static_cast<uint64_t>(mappedDataLength), request.userdata);
            return true;
        }
    }

    bool Client::DoFenceUpdateCompletedValue(Fence* fence, uint64_t value) {
        // The fence might have been deleted or recreated so this isn't an error.
        if (fence == nullptr) {
            return true;
        }

        fence->completedValue = value;
        fence->CheckPassedFences();
        return true;
    }

}}  // namespace dawn_wire::client
