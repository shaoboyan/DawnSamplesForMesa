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

#ifndef DAWNWIRE_CLIENT_DEVICE_H_
#define DAWNWIRE_CLIENT_DEVICE_H_

#include <dawn/dawn.h>

#include "dawn_wire/client/ObjectBase.h"

#include <map>

namespace dawn_wire { namespace client {

    class Client;

    class Device : public ObjectBase {
      public:
        Device(Client* client, uint32_t refcount, uint32_t id);
        ~Device();

        Client* GetClient();
        void HandleError(DawnErrorType errorType, const char* message);
        void SetUncapturedErrorCallback(DawnErrorCallback errorCallback, void* errorUserdata);

        void PushErrorScope(DawnErrorFilter filter);
        bool RequestPopErrorScope(DawnErrorCallback callback, void* userdata);
        bool PopErrorScope(uint64_t requestSerial, DawnErrorType type, const char* message);

      private:
        struct ErrorScopeData {
            DawnErrorCallback callback = nullptr;
            void* userdata = nullptr;
        };
        std::map<uint64_t, ErrorScopeData> mErrorScopes;
        uint64_t mErrorScopeRequestSerial = 0;
        uint64_t mErrorScopeStackSize = 0;

        Client* mClient = nullptr;
        DawnErrorCallback mErrorCallback = nullptr;
        void* mErrorUserdata;
    };

}}  // namespace dawn_wire::client

#endif  // DAWNWIRE_CLIENT_DEVICE_H_
