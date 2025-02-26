/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

/*
 *
 *    Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License").
 *    You may not use this file except in compliance with the License.
 *    A copy of the License is located at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    or in the "license" file accompanying this file. This file is distributed
 *    on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *    express or implied. See the License for the specific language governing
 *    permissions and limitations under the License.
 *
 */

package com.amazon.opendistroforelasticsearch.sql.opensearch.setting;

import static org.opensearch.common.settings.Settings.EMPTY;

import com.amazon.opendistroforelasticsearch.sql.common.setting.Settings;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;

/**
 * Setting implementation on OpenSearch.
 */
@Log4j2
public class OpenSearchSettings extends Settings {
  /**
   * Default settings.
   */
  private final Map<Settings.Key, Setting<?>> defaultSettings;
  /**
   * Latest setting value for each registered key. Thread-safe is required.
   */
  @VisibleForTesting
  private final Map<Settings.Key, Object> latestSettings = new ConcurrentHashMap<>();

  private static final Setting<?> PPL_QUERY_MEMORY_LIMIT_SETTINGS = Setting.memorySizeSetting(
      Key.PPL_QUERY_MEMORY_LIMIT.getKeyValue(),
      "85%",
      Setting.Property.NodeScope,
      Setting.Property.Dynamic);

  private static final Setting<?> QUERY_SIZE_LIMIT_SETTINGS = Setting.intSetting(
      Key.QUERY_SIZE_LIMIT.getKeyValue(),
      200,
      Setting.Property.NodeScope,
      Setting.Property.Dynamic);

  private static final Setting<?> PPL_ENABLED_SETTINGS = Setting.boolSetting(
      Key.PPL_ENABLED.getKeyValue(),
      true,
      Setting.Property.NodeScope,
      Setting.Property.Dynamic);

  /**
   * Construct ElasticsearchSetting.
   * The ElasticsearchSetting must be singleton.
   */
  public OpenSearchSettings(ClusterSettings clusterSettings) {
    ImmutableMap.Builder<Key, Setting<?>> settingBuilder = new ImmutableMap.Builder<>();
    register(settingBuilder, clusterSettings, Key.PPL_QUERY_MEMORY_LIMIT,
        PPL_QUERY_MEMORY_LIMIT_SETTINGS, new Updater(Key.PPL_QUERY_MEMORY_LIMIT));
    register(settingBuilder, clusterSettings, Key.PPL_ENABLED,
        PPL_ENABLED_SETTINGS, new Updater(Key.PPL_ENABLED));
    register(settingBuilder, clusterSettings, Key.QUERY_SIZE_LIMIT,
        QUERY_SIZE_LIMIT_SETTINGS, new Updater(Key.QUERY_SIZE_LIMIT));
    defaultSettings = settingBuilder.build();
  }

  @SuppressWarnings("unchecked")
  @Override
  public <T> T getSettingValue(Settings.Key key) {
    return (T) latestSettings.getOrDefault(key, defaultSettings.get(key).getDefault(EMPTY));
  }

  /**
   * Register the pair of {key, setting}.
   */
  private void register(ImmutableMap.Builder<Key, Setting<?>> settingBuilder,
                        ClusterSettings clusterSettings, Settings.Key key,
                        Setting setting,
                        Consumer<Object> updater) {
    settingBuilder.put(key, setting);
    clusterSettings
        .addSettingsUpdateConsumer(setting, updater);
  }

  /**
   * Add the inner class only for UT coverage purpuse.
   * Lambda could be much elegant solution. But which is hard to test.
   */
  @VisibleForTesting
  @RequiredArgsConstructor
  class Updater implements Consumer {
    private final Settings.Key key;

    @Override
    public void accept(Object newValue) {
      log.debug("The value of setting [{}] changed to [{}]", key, newValue);
      latestSettings.put(key, newValue);
    }
  }

  /**
   * Used by Plugin to init Setting.
   */
  public static List<Setting<?>> pluginSettings() {
    return new ImmutableList.Builder<Setting<?>>()
        .add(PPL_QUERY_MEMORY_LIMIT_SETTINGS)
        .add(PPL_ENABLED_SETTINGS)
        .add(QUERY_SIZE_LIMIT_SETTINGS)
        .build();
  }
}
