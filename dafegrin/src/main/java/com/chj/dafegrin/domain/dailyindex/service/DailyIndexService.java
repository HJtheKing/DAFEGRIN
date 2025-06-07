package com.chj.dafegrin.domain.dailyindex.service;

import com.chj.dafegrin.domain.dailyindex.repository.DailyIndexRepository;
import com.chj.dafegrin.entity.DailyIndex;
import jakarta.persistence.EntityNotFoundException;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.time.LocalDate;

@Service
@RequiredArgsConstructor
@Transactional
public class DailyIndexService {
    private final DailyIndexRepository dailyIndexRepository;
    @Value("${naver.finance.url}")
    private String URL;

    public void update(LocalDate now, Double kospi) {
        DailyIndex dailyIndex = dailyIndexRepository.findByDate(now)
                .orElseThrow(() -> new EntityNotFoundException("데이터 없음"));
        dailyIndex.setKospi(kospi);
        dailyIndexRepository.save(dailyIndex);
    }

    public Double getKospi() {
        Document doc = null;
        try {
            doc = Jsoup.connect(URL)
                    .userAgent("Mozilla/5.0")
                    .get();
            Element element = doc.selectFirst("span#change_value_and_rate");
            String percentStr = null;
            if (element != null) {
                percentStr = element.ownText();
                return Double.parseDouble(percentStr.replace("%", "").trim());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return null;
    }
}
